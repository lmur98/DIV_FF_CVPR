import os

import shutil
import argparse

# Configura argparse para aceptar un argumento de escena desde la línea de comandos
parser = argparse.ArgumentParser(description='Copy images from low res to high res folder.')
parser.add_argument('scene', type=str, help='The scene to process, e.g., P01_01')
args = parser.parse_args()

dataset_path = '/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data/EPIC-Diff'
scene = args.scene
frames_path = os.path.join(dataset_path, scene, 'frames')
low_res_imgs = [img for img in os.listdir(frames_path) if img.endswith('.bmp')]
images_ID = [img.split('.')[0] for img in low_res_imgs]
print(len(images_ID))

imgs_to_copy = [img.replace("IMG", "frame") for img in images_ID]
imgs_to_copy = [img + '.jpg' for img in imgs_to_copy]
imgs_to_copy = [os.path.join(dataset_path, scene, 'high_res_video', img) for img in imgs_to_copy]
print(len(imgs_to_copy))

#Copy images to a new folder, called images
import shutil
new_folder = os.path.join(dataset_path, scene, 'images')
os.makedirs(new_folder, exist_ok=True)
for img in imgs_to_copy:
    print(img)
    shutil.copy(img, new_folder)