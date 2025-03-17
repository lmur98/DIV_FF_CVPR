import os
import random
import argparse

import torch
import torchvision
from torch import nn
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only make GPU:1 visible

import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

import sys

# Establecer PYTHONPATH dentro del script Python
sys.path.append("/home/lmur/FUSION_FIELDS/sam2")


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__() #Antes era bfloat16
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sys.path.append('/home/lmur/FUSION_FIELDS')
from EgoVideo.backbone.model.setup_model import build_video_model, build_text_model
from matplotlib.colors import ListedColormap, Normalize

def extract_bbox_from_mask(mask):
    mask_ = mask.squeeze()

    if not np.any(mask_):
        return None  

    # Obtener las coordenadas de los valores True en la máscara
    coords = np.argwhere(mask_)

    # Extraer las coordenadas mínimas y máximas de la bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Retornar la bounding box en el formato [x_min, y_min, x_max, y_max]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def extract_tublet_bbox(bbox_dict):
    # Inicializar con valores extremos
    x_min_union = float('inf')
    y_min_union = float('inf')
    x_max_union = float('-inf')
    y_max_union = float('-inf')
    
    # Recorrer todas las bounding boxes en el diccionario
    for key, bbox in bbox_dict.items():
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        # Actualizar los valores extremos
        x_min_union = min(x_min_union, x_min)
        y_min_union = min(y_min_union, y_min)
        x_max_union = max(x_max_union, x_max)
        y_max_union = max(y_max_union, y_max)
    return [x_min_union, y_min_union, x_max_union, y_max_union]

def extract_video_crop(video_clip, tublet_bbox, transform):
    cropped_clips = []
    for b in range(tublet_bbox.shape[0]):
        x_min, y_min, x_max, y_max = tublet_bbox[b]
        cropped_part = video_clip[:, :, y_min:y_max, x_min:x_max]
        cropped_clips.append(transform(cropped_part))
    return torch.stack(cropped_clips, dim=0)

def mask2segmap(masks):
    seg_map = -np.ones(masks[0]['segmentation'].shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_map[masks[i]['segmentation']] = i
    return seg_map

def save_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'])
    np.save(save_path_f, data['feature'])
    
def paint_seg_map(seg_map, show_path):
    seg_map = seg_map.copy()
    unique_classes = np.unique(seg_map)
    N_classes = len(np.unique(seg_map))
    colors = np.random.rand(N_classes, 3)
    colors[0] = [0, 0, 0]
    
    class_map = {k: i for i, k in enumerate(unique_classes)}
    seg_map_mapped = np.vectorize(class_map.get)(seg_map)
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(seg_map_mapped, cmap=cmap, norm=Normalize(vmin=0, vmax=N_classes-1))
    plt.title('Segmentation map of the last frame')
    plt.savefig(show_path + '_seg_map.png')

def create(videos_list, save_folder, transform, number_frames = 4):
    for i, video in tqdm(enumerate(videos_list), desc="Processing videos with SAM", leave=False):
        frame_ID = video['frame_ID']
        frames_paths = video['video_frames']
        video_dir = video['video_dir']
        
        # Using OpenCV to read the last frame
        last_frame_image = cv2.imread(frames_paths[0])  # This reads the image as a NumPy array
        last_frame_image = cv2.cvtColor(last_frame_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed for display
        
        video_clip = []
        for f in frames_paths[::-1]: #Cambiamos el orden de los frames para que el último sea el primero
            video_clip.append(torchvision.transforms.functional.to_tensor(Image.open(f)))
        video_clip = torch.stack(video_clip, dim=1)
        
        # Obtain masks of the last frame in order to initialize the mask generator
        masks_last_frame = mask_image_generator.generate(last_frame_image)  
        ann_obj_id = [i for i in range(len(masks_last_frame))]
        points_obj = [mask['point_coords'] for mask in masks_last_frame]
        labels_obj = [np.array([1], dtype=np.int32) for mask in masks_last_frame]  
        seg_map = mask2segmap(masks_last_frame)

        bbox_tiles = []
        for ann_obj_id, points, labels in zip(ann_obj_id, points_obj, labels_obj):
            inference_state = video_predictor.init_state(video_path=video_dir) 
            #HEMOS MODIFICADO EL SAM2 -> UTILS -> MISC
            ann_frame_idx = 0
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                video_segments[number_frames - 1 - out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            #Video segments is a dictionary with the frame index as key. The value is another dictionary with the object ID as key and the mask as value
            bboxes_to_crop = {}
            for out_frame_idx in range(number_frames):
                mask = video_segments[out_frame_idx][ann_obj_id]
                bboxes_to_crop[out_frame_idx] = extract_bbox_from_mask(mask)
            tublet_bbox = extract_tublet_bbox(bboxes_to_crop)
            bbox_tiles.append(tublet_bbox)
        bbox_tiles = torch.stack([torch.tensor(bbox) for bbox in bbox_tiles], dim=0)
        
        #Now, we have:
        # -Masks of the last frame: where we will asign the features. They are saved as _m.sava_numpy
        # -Tiles_bbox: the bounding boxes of the tiles to crop from the video
        # -video_dir: the path to the video frames
        video_batch_size = 32
        video_tiles = extract_video_crop(video_clip, bbox_tiles, transform) #(B, C, T, H, W)
        
        for i in range(0, video_tiles.shape[0], video_batch_size):
            video_tiles_batch = video_tiles[i:i+video_batch_size].to(device)
            with torch.no_grad():
                video_embeds, _ = EgoVideo_model(video_tiles_batch, None, None)
            video_embeds = video_embeds.detach().cpu().numpy()
            if i == 0:
                features = video_embeds
            else:
                features = np.concatenate((features, video_embeds), axis=0)
        
        
        curr = {'feature': features, 'seg_maps': seg_map}
        save_path = os.path.join(save_folder, frame_ID)
        save_numpy(save_path, curr)
        paint_seg_map(seg_map, save_path)


if __name__ == '__main__':
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


    seed_num = 42
    random.seed(seed_num)

    root_path = '/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data/EPIC-Diff/'
    scene = 'P01_01'
    
    img_folder = os.path.join(root_path, scene, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    #We define EgoVideo for extracting the features
    video_encoder, _ = build_video_model(ckpt_path = '/home/lmur/FUSION_FIELDS/EgoVideo/ckpt_4frames.pth', num_frames = 4)
    EgoVideo_model = video_encoder.eval().to("cuda:0") #.to(torch.float16)
    video_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                      torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),])
    
    
    #SAM models
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    image_predictor = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    #mask_image_generator = SAM2AutomaticMaskGenerator(image_predictor)
    
    mask_image_generator = SAM2AutomaticMaskGenerator(
        model=image_predictor,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=225,
    )

    
    #We define the video parameters
    temporal_stride = 15 #This covers 1 seconds of video, since our frames are at 60 fps
    sampled_frames = 4
    high_res_video_path = os.path.join(root_path, scene, 'high_res_video')

    videos_list = []
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        frame_id = image_path.split('/')[-1].split('_')[-1].split('.')[0]
        frame_number = int(frame_id)
        #We take the previous 2 secs of video
        video_frames = [frame_number - i for i in range(0, sampled_frames * temporal_stride, temporal_stride)]
        video_frames_paths = []
        for frame in video_frames:
            # Creamos un nuevo filename con el número de frame formateado a 10 dígitos
            new_filename = f"frame_{str(frame).zfill(10)}.jpg"
            new_frame_path = os.path.join(high_res_video_path, new_filename)
            video_frames_paths.append(new_frame_path)
        
        #Create a folder to save the video frames
        video_dir = os.path.join(high_res_video_path, 'clip_' + frame_id)
        os.makedirs(video_dir, exist_ok=True)
        for frame in video_frames_paths:
            frame_name = frame.split('/')[-1]
            os.system(f'cp {frame} {os.path.join(high_res_video_path, "clip_" + frame_id, frame_name)}')
        videos_list.append({'frame_ID': frame_id, 'video_frames': video_frames_paths, 'video_dir': video_dir})

save_folder = os.path.join('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/outputs', scene, 'SAM_EgoVIDEO_features')
os.makedirs(save_folder, exist_ok=True)
create(videos_list, save_folder, video_transform)