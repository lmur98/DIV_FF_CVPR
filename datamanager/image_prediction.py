import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax, color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))  

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

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        #plt.show()
        plt.savefig('/home/lmur/FUSION_FIELDS/sam2/video_prueba/mask_'+ str(i) +'_with_point.png')

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

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
  


from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


video_dir = "/home/lmur/FUSION_FIELDS/sam2/video_prueba/kitchen_video"
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]), reverse=True)  # Adjusted to split by '_' and convert the last part to int
print(frame_names, 'los frame numbeeeeeeerrssss')

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
image_predictor = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_image_generator = SAM2AutomaticMaskGenerator(image_predictor)

# Using OpenCV to read the last frame
last_frame_path = os.path.join(video_dir, frame_names[0])
last_frame_image = cv2.imread(last_frame_path)  # This reads the image as a NumPy array
last_frame_image = cv2.cvtColor(last_frame_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed for display

# Generate masks
masks_last_frame = mask_image_generator.generate(last_frame_image)  
plt.figure(figsize=(20, 20))
plt.imshow(Image.open(os.path.join(video_dir, frame_names[0])))
show_anns(masks_last_frame)
plt.axis('off')
plt.savefig('/home/lmur/FUSION_FIELDS/sam2/video_prueba/results/masks_last_frame.png')

ann_obj_id = [i for i in range(len(masks_last_frame))]
points_obj = [mask['point_coords'] for mask in masks_last_frame]
labels_obj = [np.array([1], dtype=np.int32) for mask in masks_last_frame]   
# (mask['point_coords'], np.array(i, dtype=np.int32)) for i, mask in enumerate(masks_last_frame)}
# Segment each frame using the prompts from the last frame

number_of_frames = 4
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
        video_segments[number_of_frames - 1 - out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    #Video segments is a dictionary with the frame index as key. The value is another dictionary with the object ID as key and the mask as value

    bboxes = {}
    for out_frame_idx in range(number_of_frames):
        mask = video_segments[out_frame_idx][ann_obj_id]
        bboxes[out_frame_idx] = extract_bbox_from_mask(mask)
    tublet_bbox = extract_tublet_bbox(bboxes)
    bboxes['object_id'] = ann_obj_id
    bboxes['tublet_bbox'] = tublet_bbox
    print(f"Segmentation for frame {ann_frame_idx} done")
    print('video_segments', video_segments)
    
    plt.close('all')
    for out_frame_idx in range(4):
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {out_frame_idx}")
        print(os.path.join(video_dir, frame_names[out_frame_idx]), 'queeee abre')
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, mask in video_segments[out_frame_idx].items():
            show_mask(mask, plt.gca(), obj_id=out_obj_id)
            if bboxes[out_frame_idx] is not None:
                show_box(bboxes[out_frame_idx], plt.gca())
            if bboxes['tublet_bbox'] is not None:
                show_box(bboxes['tublet_bbox'], plt.gca(), color='red')
            plt.axis('off')
            plt.savefig('/home/lmur/FUSION_FIELDS/sam2/video_prueba/results/objectID_' + str(out_obj_id) + '_frame_id_' + str(out_frame_idx) + '.png')
    print()
