"""Render a summary video as shown on the project page."""
import math

import numpy as np
import torch

from . import segmentation, utils
import cv2


def render(dataset, model, sample_id=None, n_images=20):
    """
    Render a video for a dataset and model.
    If a sample_id is selected, then the view is fixed and images
    are rendered for a specific viewpoint over a timerange (the bottom part
    of the summary video on the project page). Otherwise, images are rendered
    for multiple viewpoints (the top part of the summary video).
    """

    ims = {}

    keys = [
        "mask_pers",
        "mask_tran",
        "mask_pred",
        "im_tran",
        "im_stat",
        "im_pred",
        "im_pers",
        "im_targ",
    ]

    for i in range(len(dataset.image_encoder.positives)):
        keys.append(f"composited_{i}")

    if n_images > len(dataset.img_ids) or n_images == 0:
        n_images = len(dataset.img_ids)

    for i in utils.tqdm(dataset.img_ids[:: math.ceil(len(dataset.img_ids) / n_images)]):
        if sample_id is not None:
            j = sample_id
        else:
            j = i
        timestep = i
        with torch.no_grad():
            x = segmentation.evaluate_sample(
                dataset, j, t=timestep, model=model, visualise=False
            )
            ims[i] = {k: x[k] for k in x if k in keys}
    return ims


def cat_sample(top, bot=None, keys=["im_targ", "im_pred", "im_stat", "im_tran", "im_pers"], keys_text = None):
    """Concatenate images from the top and bottom part of the summary video."""
    top = np.concatenate([(top[k]) for k in keys], axis=1)
    if (bot is not None):
        bot = np.concatenate([(bot[k]) for k in keys], axis=1)
        bot[
            :,
            : bot.shape[1] // len(keys),  # black background in first column
        ] = (0, 0, 0)
        z = np.concatenate([top, bot], axis=0)
    else:
        z = top
        
    #Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    fontcolor = (255, 255, 255)
    thickness = 1
    height = 80
    bottom_text = []
    for key in keys_text:
        text_image = np.zeros((height, 228, 3), dtype=np.uint8)
        (text_width, text_height) = cv2.getTextSize("Target", font, fontScale=fontScale, thickness=thickness)[0]
        x = (text_image.shape[1] - text_width) // 3
        y = (height + text_height) // 2
        cv2.putText(text_image, str(key), (x, y), font, fontScale, fontcolor, thickness=thickness)
        bottom_text.append(text_image)
    bottom_text = np.concatenate(bottom_text, axis=1)
    z = np.concatenate([z, bottom_text], axis=0)
    
    return z


def save_to_cache(vid, sid, root, top=None, bot=None):
    """Save the images for rendering the video."""
    if top is not None:
        p = f"{root}/images-{sid}-top.pt"
        if os.path.exists(p):
            print("images exist, aborting.")
            return
        torch.save(top, p)
    if bot is not None:
        p = f"{root}/images-{sid}-bot.pt"
        if os.path.exists(p):
            print("images exist, aborting.")
            return
        torch.save(bot, p)


def load_from_cache(vid, sid, root, version=0):
    """Load the images for rendering the video."""
    path_top = f"{root}/images-{sid}-top.pt"
    path_bot = f"{root}/images-{sid}-bot.pt"
    top = torch.load(path_top)
    bot = torch.load(path_bot)
    return top, bot


def convert_rgb(im):
    im[im > 1] = 1
    im = (im * 255).astype(np.uint8)
    return im
