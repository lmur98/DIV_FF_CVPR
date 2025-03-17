import torch
from einops import rearrange, reduce, repeat


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / reduce(weights, "n1 n2 -> n1 1", "sum")
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    u = torch.rand(N_rays, N_importance, device=bins.device)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(
        torch.stack([below, above], -1), "n1 n2 c -> n1 (n2 c)", c=2
    )
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)

    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples

def get_max_across(models, image_encoder, encoding_samples, weights, scales_shape, preset_scales=None):
    if preset_scales is not None:
        assert len(preset_scales) == len(image_encoder.positives)
        scales_list = preset_scales.float() # torch.tensor(preset_scales)
    else:
        scales_list = torch.linspace(0.0, 1.5, 30)

    parts = ['static', 'transient', 'person'] #, 'transient', 'person']
    
    # probably not a good idea bc it's prob going to be a lot of memory
    n_phrases = len(image_encoder.positives)
    n_phrases_maxs = [None for _ in range(n_phrases)]
    n_phrases_sims = [None for _ in range(n_phrases)]
    for i, scale in enumerate(scales_list):
        scale = scale.item()
        clip_out_list = []
        for p in parts:
            with torch.no_grad():
                clip_output = models[p + '_lerf'].get_output_from_hashgrid(
                    encoding_samples[p],
                    torch.full(scales_shape, scale, device=weights[p].device, dtype=encoding_samples[p].dtype),
                )
            clip_output = torch.sum(weights[p].detach() * clip_output, dim=-2)
            #NO!! falla al hacer el suppress person ##clip_output = torch.nn.functional.normalize(clip_output, dim = -1) #clip_output / torch.linalg.norm(clip_output, dim=-1, keepdim=True)
            clip_out_list.append(clip_output)
        clip_output = sum(clip_out_list) #torch.nn.functional.normalize(sum(clip_out_list), dim = -1)
            
        for j in range(n_phrases):
            if preset_scales is None or j == i:
                probs = image_encoder.get_relevancy(clip_output, j)
                pos_prob = probs[..., 0:1]
                if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                    n_phrases_maxs[j] = scale
                    n_phrases_sims[j] = pos_prob
    return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

def render_rays(
    models,
    embeddings,
    rays,
    ts,
    clip_scales,
    img_h,
    intrinsics_fy,
    N_samples=64,
    perturb=0,
    noise_std=1,
    N_importance=0,
    N_lerf_samples=12,
    chunk=1024 * 32,
    test_time=False,
    image_encoder=None,
    **kwargs,
):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of models (coarse and fine)
        embeddings: dict of embedding models of origin and direction
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays), ray time as embedding index
        N_samples: number of coarse samples per ray
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        results: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, xyz, z_vals, test_time=False, xyz_c=None, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: coarse or fine model
            xyz: (N_rays, N_samples_, 3) sampled positions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
            xyz_c: (N_rays, N_samples_, 3) sampled positions w.r.t. camera coordinates
        """
        typ = model.typ
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c", c=3)
        xyz_c_ = rearrange(xyz_c, "n1 n2 c -> (n1 n2) c", c=3)

        # Perform model inference to get rgb, sigma
        B = xyz_.shape[0]
        out_chunks = []
        xyz_c_embedded_ = embedding_xyz(xyz_c_)
        if typ == "coarse" and test_time:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i : i + chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]
            out = torch.cat(out_chunks, 0)
            static_sigmas = rearrange(
                out, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_
            )
        else:
            #
            dir_embedded_ = repeat(dir_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
            # create other necessary inputs
            if output_dynamic:
                a_embedded_ = repeat(a_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
                t_embedded_ = repeat(t_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
            for i in range(0, B, chunk):
                # inputs for original NeRF
                inputs = [
                    embedding_xyz(xyz_[i : i + chunk]),
                    dir_embedded_[i : i + chunk],
                ]
                # additional inputs
                if output_dynamic:
                    inputs += [a_embedded_[i : i + chunk]]
                    inputs += [t_embedded_[i : i + chunk]]
                    inputs += [embedding_xyz(xyz_c_[i : i + chunk])]
                if test_time:
                    out_chunks += [
                        model(
                            torch.cat(inputs, 1),
                            output_dynamic=output_dynamic,
                        ).half()
                    ]
                else:
                    out_chunks += [
                        model(
                            torch.cat(inputs, 1),
                            output_dynamic=output_dynamic,
                        )
                    ]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_)
            static_rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
            static_sigmas = out[..., 3]  # (N_rays, N_samples_)
            if output_dynamic:         
                transient_rgbs = out[..., 4:7]
                transient_sigmas = out[..., 7]
                transient_betas = out[..., 8]
                person_rgbs = out[..., 9:12]
                person_sigmas = out[..., 12]
                person_betas = out[..., 13]

                enc_size = 128
                static_encoding = out[..., 14:14 + enc_size]
                results["static_encoding"] = static_encoding
                transient_encoding = out[..., 14 + enc_size:14 + 2 * enc_size]
                results["transient_encoding"] = transient_encoding
                person_encoding = out[..., 14 + 2 * enc_size:14 + 3 * enc_size]
                results["person_encoding"] = person_encoding

                #print(hp.inference, hp.suppress_person)
                #if hp.suppress_person:
                    # disables person during inference, e.g. for visualising videos
            #person_sigmas[:] = 0
            #transient_sigmas[:] = 0
            #static_sigmas[:] = 0
                
            if test_time:
                n_channels = 1 + output_dynamic * 2
                stat = torch.zeros([*static_rgbs.shape[:2], n_channels]).to(
                    static_rgbs.device
                )
                stat[:, :, 0] = 1

                static_rgbs = torch.cat([static_rgbs, stat], dim=2)
                if output_dynamic:
                    tran = torch.zeros([*static_rgbs.shape[:2], n_channels]).to(
                        static_rgbs.device
                    )
                    tran[:, :, 1] = 1
                    transient_rgbs = torch.cat([transient_rgbs, tran], dim=2)
                    pers = torch.zeros([*static_rgbs.shape[:2], n_channels]).to(
                        static_rgbs.device
                    )
                    pers[:, :, 2] = 1
                    person_rgbs = torch.cat([person_rgbs, pers], dim=2)

        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1])
        # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # add RGB noise to last segments of rays to avoid trivial rendering of "black" colours (0,0,0)
        # "transmittance fix"
        if not typ == "coarse":
            if test_time:
                pass
            else:
                static_sigmas[:, -1:] = 100
                static_rgbs[:, -1, :3] = torch.rand(static_rgbs.shape[0], 3).to(
                    static_rgbs.device
                )

        if output_dynamic:
            # for colour normalisation as described in "Improved color mixing"
            sum_sigmas = static_sigmas + transient_sigmas + person_sigmas
            alphas = 1 - torch.exp(-deltas * (sum_sigmas))

            # ignore normalisation for last value to stabilise inf delta (described above)
            static_alphas = static_sigmas / (sum_sigmas + 1e-6) * alphas #
            static_alphas[:, -1:] = (1 - torch.exp(-deltas * static_sigmas))[:, -1:]
            transient_alphas = transient_sigmas / (sum_sigmas + 1e-6) * alphas
            transient_alphas[:, -1:] = (1 - torch.exp(-deltas * transient_sigmas))[
                :, -1:
            ]
            person_alphas = person_sigmas / (sum_sigmas + 1e-6) * alphas
            person_alphas[:, -1:] = (1 - torch.exp(-deltas * person_sigmas))[:, -1:]

            #results[f"static_alphas_{typ}"] = static_alphas
            #results[f"transient_alphas_{typ}"] = transient_alphas
            #results[f"person_alphas_{typ}"] = person_alphas
        else:
            noise = torch.randn_like(static_sigmas) * noise_std
            alphas = 1 - torch.exp(-deltas * torch.relu(static_sigmas + noise))

        results[f"alphas_{typ}"] = alphas

        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas], -1
        )  # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(
            alphas_shifted[:, :-1], -1
        )  # [1, 1-a1, (1-a1)(1-a2), ...]

        if not (typ == "coarse" and test_time):
            results[f"static_rgbs_{typ}"] = static_rgbs

        #results[f"transmittance_{typ}"] = transmittance

        if output_dynamic:
            static_weights = static_alphas * transmittance
            results[f"static_weights"] = static_weights
            results[f"static_alphas_{typ}"] = static_alphas
            transient_weights = transient_alphas * transmittance
            transient_weights_sum = reduce(transient_weights, "n1 n2 -> n1", "sum")
            person_weights = person_alphas * transmittance
            person_weights_sum = reduce(person_weights, "n1 n2 -> n1", "sum")

        weights = alphas * transmittance
        weights_sum = reduce(weights, "n1 n2 -> n1", "sum")

        results[f"weights_{typ}"] = weights
        results[f"static_weights_{typ}"] = weights
        results[f"static_weights_sum_{typ}"] = weights_sum
        #results[f"opacity_{typ}"] = weights_sum
        results[f"static_sigmas_{typ}"] = static_sigmas
        if output_dynamic:
            results["transient_sigmas"] = transient_sigmas
            results["transient_weights"] = transient_weights
            results["transient_weights_sum"] = transient_weights_sum
            results["person_sigmas"] = person_sigmas
            results["person_weights"] = person_weights
            results["person_weights_sum"] = person_weights_sum
        if test_time and typ == "coarse":
            return

        if output_dynamic:
            static_rgb_map = reduce(
                rearrange(static_weights, "n1 n2 -> n1 n2 1") * static_rgbs,
                "n1 n2 c -> n1 c",
                "sum",
            )

            transient_rgb_map = reduce(
                rearrange(transient_weights, "n1 n2 -> n1 n2 1") * transient_rgbs,
                "n1 n2 c -> n1 c",
                "sum",
            )
            results["beta"] = reduce(
                transient_weights * transient_betas, "n1 n2 -> n1", "sum"
            )

            # the rgb maps here are when both fields exist
            results["_rgb_fine_static"] = static_rgb_map
            results["_rgb_fine_transient"] = transient_rgb_map
            results["rgb_fine"] = static_rgb_map + transient_rgb_map

            person_rgb_map = reduce(
                rearrange(person_weights, "n1 n2 -> n1 n2 1") * person_rgbs,
                "n1 n2 c -> n1 c",
                "sum",
            )
            results["beta"] = results["beta"] + reduce(
                person_weights * person_betas, "n1 n2 -> n1", "sum"
            )

            # the rgb maps here are when both fields exist
            results["_rgb_fine_person"] = person_rgb_map
            results["rgb_fine"] = results["rgb_fine"] + person_rgb_map

            results["beta"] += model.beta_min

            if test_time:
                static_alphas_shifted = torch.cat(
                    [torch.ones_like(static_alphas[:, :1]), 1 - static_alphas], -1
                )
                static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
                static_weights_ = static_alphas * static_transmittance
                static_rgb_map_ = reduce(
                    rearrange(static_weights_, "n1 n2 -> n1 n2 1") * static_rgbs,
                    "n1 n2 c -> n1 c",
                    "sum",
                )
                results["rgb_fine_static"] = static_rgb_map_
                results["depth_fine_static"] = reduce(
                    static_weights_ * z_vals, "n1 n2 -> n1", "sum"
                )

                transient_alphas_shifted = torch.cat(
                    [torch.ones_like(transient_alphas[:, :1]), 1 - transient_alphas], -1
                )
                transient_transmittance = torch.cumprod(
                    transient_alphas_shifted[:, :-1], -1
                )
                transient_weights_ = transient_alphas * transient_transmittance
                results["rgb_fine_transient"] = reduce(
                    rearrange(transient_weights_, "n1 n2 -> n1 n2 1") * transient_rgbs,
                    "n1 n2 c -> n1 c",
                    "sum",
                )
                results["depth_fine_transient"] = reduce(
                    transient_weights_ * z_vals, "n1 n2 -> n1", "sum"
                )

                person_alphas_shifted = torch.cat(
                    [torch.ones_like(person_alphas[:, :1]), 1 - person_alphas], -1
                )
                person_transmittance = torch.cumprod(
                    person_alphas_shifted[:, :-1], -1
                )
                person_weights_ = person_alphas * person_transmittance
                results["rgb_fine_person"] = reduce(
                    rearrange(person_weights_, "n1 n2 -> n1 n2 1") * person_rgbs,
                    "n1 n2 c -> n1 c",
                    "sum",
                )
                results["depth_fine_person"] = reduce(
                    person_weights_ * z_vals, "n1 n2 -> n1", "sum"
                )

        else:  # no transient field
            rgb_map = reduce(
                rearrange(weights, "n1 n2 -> n1 n2 1") * static_rgbs,
                "n1 n2 c -> n1 c",
                "sum",
            )
            results[f"rgb_{typ}"] = rgb_map

        results[f"depth_{typ}"] = reduce(weights * z_vals, "n1 n2 -> n1", "sum")
        return
    
    def lerf_inference_VIDEO(results, models, encoding, weights, test_time=False, **kwargs):
        """
        Helper function that performs LERF inference.
        Inputs:
            results: a dict storing all results
            model: lerf model
            xyz: (N_rays, N_lerf_samples, 3) sampled positions 
            scales: (N_rays, N_lerf_samples, 1) clip scales for each ray
            test_time: test time or not
            preset_scales: (N_positives) overrided scales
        """
        parts = ['static', 'transient', 'person'] 
        video_out_list = []
        dino_out_list = []
        for p in parts:
            video_out, dino_out = models[p + '_video'](encoding[p])
            video_out = torch.sum(weights[p] * video_out, dim=-2) #MOD: previously it was detach() 
            dino_out = torch.sum(weights[p] * dino_out, dim=-2)
            #NO!! falla al hacer el suppress person #clip_out = torch.nn.functional.normalize(clip_out, dim = -1) #clip_out / torch.linalg.norm(clip_out, dim=-1, keepdim=True) #THIS WAY IS MUCH MORE STABLE
            video_out_list.append(video_out)
            dino_out_list.append(dino_out)
            
        video_output = torch.nn.functional.normalize(sum(video_out_list), dim = -1)
        results["video"] = video_output
        dino_output = torch.nn.functional.normalize(sum(dino_out_list), dim = -1)
        results["dino"] = dino_output


        """
        #hash_out, clip_out, dino_out = model(xyz, scales)
        video_out, dino_out = models["video_lerf"](encoding)
        video_out = torch.sum(weights * video_out, dim=-2)
        video_out = torch.nn.functional.normalize(video_out, dim = -1) #Normalize for the posterior computation of the similarity
        results["video"] = video_out
        dino_out = torch.sum(weights * dino_out, dim=-2) 
        dino_out = torch.nn.functional.normalize(dino_out, dim = -1) #Normalize for the posterior computation of the similarity
        results["dino"] = dino_out
        """

        if test_time:
            with torch.no_grad():
                #video_out_rescaled = image_encoder.VIDEO_SAM_autoencoder.decode(video_out)
                video_out_rescaled = results["video"]
                
                n_phrases = len(image_encoder.video_positives)

                n_phrases_maxs = [None for _ in range(n_phrases)]
                n_phrases_sims = [None for _ in range(n_phrases)]
                for j in range(n_phrases):
                    #if preset_scales is None or j == i:
                    probs = image_encoder.get_video_relevancy(video_out_rescaled, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_sims[j] = pos_prob
                max_across = torch.stack(n_phrases_sims)
            results["raw_video_relevancy"] = max_across # [N_positives, N_Rays, 1] 
        
        return
    
    def lerf_inference_SINGLE_SCALE(results, models, encoding, weights, test_time=False, **kwargs):
        """
        Helper function that performs LERF inference.
        Inputs:
            results: a dict storing all results
            model: lerf model
            xyz: (N_rays, N_lerf_samples, 3) sampled positions 
            scales: (N_rays, N_lerf_samples, 1) clip scales for each ray
            test_time: test time or not
            preset_scales: (N_positives) overrided scales
        """
        #hash_out, clip_out, dino_out = model(xyz, scales)
        parts = ['static', 'transient', 'person'] 
        clip_out_list = []
        for p in parts:
            clip_out = models[p + '_lerf'](encoding[p])
            clip_out = torch.sum(weights[p] * clip_out, dim=-2) #MOD: previously it was detach() 
            #NO!! falla al hacer el suppress person #clip_out = torch.nn.functional.normalize(clip_out, dim = -1) #clip_out / torch.linalg.norm(clip_out, dim=-1, keepdim=True) #THIS WAY IS MUCH MORE STABLE
            clip_out_list.append(clip_out)

        clip_output = torch.nn.functional.normalize(sum(clip_out_list), dim = -1)
        results["clip"] = clip_output
        

        if test_time:
            with torch.no_grad(): 
                # GT MAX ACROSS FUNCTION
                clip_output_rescaled = image_encoder.IMAGE_SAM_autoencoder.decode(clip_output)
                n_phrases = len(image_encoder.positives)
                n_phrases_maxs = [None for _ in range(n_phrases)]
                n_phrases_sims = [None for _ in range(n_phrases)]
                for j in range(n_phrases):
                    #if preset_scales is None or j == i:
                    probs = image_encoder.get_relevancy(clip_output_rescaled, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_sims[j] = pos_prob
                max_across = torch.stack(n_phrases_sims)
                results["raw_relevancy"] = max_across # [N_positives, N_Rays, 1] 

        return

    hp = kwargs["hp"]

    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]

    # separate input into ray origins, directions, near, far bounds etc.
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]
    rays_o_c, rays_d_c = rays[:, 8:11], rays[:, 11:14]
    dir_embedded = embedding_dir(kwargs.get("view_dir", rays_d))

    rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
    rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

    rays_o_c = rearrange(rays_o_c, "n1 c -> n1 1 c")
    rays_d_c = rearrange(rays_d_c, "n1 c -> n1 1 c")

    # sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    z_vals = near * (1 - z_steps) + far * z_steps

    z_vals = z_vals.expand(N_rays, N_samples)

    # perturb sampling depths (z_vals)
    perturb_rand = perturb * torch.rand_like(z_vals)
    if perturb > 0:
        # (N_rays, N_samples-1) interval mid points
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        z_vals = lower + (upper - lower) * perturb_rand

    results = {}
    xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
    xyz_coarse_c = rays_o_c + rays_d_c * rearrange(z_vals, "n1 n2 -> n1 n2 1")
    # disable transient and person model for coarse model
    output_dynamic = False
    inference(
        results,
        models["coarse"],
        xyz_coarse,
        z_vals,
        test_time,
        xyz_c=xyz_coarse_c,
        **kwargs,
    )

    # then continue with fine model by sampling from z_vals of coarse model
    if N_importance > 0:
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(
            z_vals_mid,
            results["weights_coarse"][:, 1:-1].detach(),
            N_importance,
            det=(perturb == 0),
        )
        # detach so that grad doesn't propogate to weights_coarse from here
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        xyz_fine = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
        xyz_fine_c = rays_o_c + rays_d_c * rearrange(z_vals, "n1 n2 -> n1 n2 1")

        model = models["fine"]
        output_dynamic = True
        t_embedded = embeddings["t"](ts)
        a_embedded = embeddings["a"](ts)
        inference(
            results, model, xyz_fine, z_vals, test_time, xyz_c=xyz_fine_c, **kwargs
        )
        
        if hp.use_clip:
            #STATIC MODEL
            # Sample positions from xyz_fine for LERF model
            static_weights, best_ids = torch.topk(rearrange(results["static_weights"], "n1 n2 -> n1 n2 1"), N_lerf_samples, dim=-2, sorted=False) # [N_Rays, N_lerf_samples, 1]
            static_encoding_lerf = torch.gather(results["static_encoding"], -2, best_ids.expand(*best_ids.shape[:-1], 128))

            #TRANSIENT MODEL
            transient_weights, best_ids = torch.topk(rearrange(results["transient_weights"], "n1 n2 -> n1 n2 1"), N_lerf_samples, dim=-2, sorted=False) # [N_Rays, N_lerf_samples, 1]
            transient_encoding_lerf = torch.gather(results["transient_encoding"], -2, best_ids.expand(*best_ids.shape[:-1], 128))
            
            #PERSON MODEL
            person_weights, best_ids = torch.topk(rearrange(results["person_weights"], "n1 n2 -> n1 n2 1"), N_lerf_samples, dim=-2, sorted=False)
            person_encoding_lerf = torch.gather(results["person_encoding"], -2, best_ids.expand(*best_ids.shape[:-1], 128))

            encoding_lerf = {'static': static_encoding_lerf, 'transient': transient_encoding_lerf, 'person': person_encoding_lerf}
            weights = {'static': static_weights,  'transient': transient_weights, 'person': person_weights}
            
            
            lerf_inference_SINGLE_SCALE(
                results,
                models,
                encoding_lerf, #MOD!! Before it was xyz_lerf
                weights,
                test_time
            )
            if hp.video_active:
                #video_weights, best_ids = torch.topk(rearrange(results["weights_fine"], "n1 n2 -> n1 n2 1"), N_lerf_samples, dim=-2, sorted=False)
                #video_encoding_lerf = torch.gather(xyz_fine, -2, best_ids.expand(*best_ids.shape[:-1], 3))
                
                lerf_inference_VIDEO(
                    results,
                    models,
                    encoding_lerf, #video_encoding_lerf, #MOD!! Before it was xyz_lerf
                    weights, #video_weights,
                    test_time,
                )

            if test_time:
                # Re-render max across using best scales form first render
                #lerf_inference_SINGLE_SCALE(
                #    results,
                #    models,
                #    encoding_lerf, #MOD!! Before it was xyz_lerf
                #    weights,
                #    test_time,
                #)        
                for r_id in range(results["raw_relevancy"].shape[0]):
                    results[f"relevancy_{r_id}"] = (results["raw_relevancy"][r_id, ...])  
                for r_id in range(results["raw_video_relevancy"].shape[0]):
                    results[f"video_relevancy_{r_id}"] = (results["raw_video_relevancy"][r_id, ...])  

    return results
