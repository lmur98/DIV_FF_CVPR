# Dynamic Image-Video Feature Fields (DIV-FF)

## ✨✨ Accepted in CVPR 2025!! ✨✨
![🎥 Ver video](P13_03-_online-video-cutter.com_.gif)

## About
This repository contains the official implementation of the paper *Dynamic Image-Video Feature Fields (DIV-FF)*, Published at CVPR 2021

Environment understanding in egocentric videos is an important step for applications like robotics, augmented reality and assistive technologies. These videos are characterized by dynamic interactions and a strong dependence on the wearer engagement with the environment. Traditional approaches often focus on isolated clips or fail to integrate rich semantic and geometric information, limiting scene comprehension. We introduce Dynamic Image-Video Feature Fields (DIV-FF), a framework that decomposes the egocentric scene into persistent, dynamic, and actor-based components while integrating both image and video-language features. Our model enables detailed segmentation, captures affordances, understands the surroundings and maintains consistent understanding over time. DIV-FF outperforms state-of-the-art methods, particularly in dynamically evolving scenarios, demonstrating its potential to advance long-term, spatio-temporal scene understanding.

## Dataset

Download EPIC-Diff from the [original repository](https://www.robots.ox.ac.uk/~vadim/neuraldiff/release/EPIC-Diff-annotations.tar.gz). 
After downloading, move the compressed dataset to the directory of the cloned repository (e.g. NeuralDiff). Then, apply following commands:
```bash
mkdir data
mv EPIC-Diff.tar.gz data
cd data
tar -xzvf EPIC-Diff.tar.gz


## Training

Con el siguiente script puedes entrenar la parte de geometría (NeRFs) sin activar CLIP/DINO (puedes activarlo si quieres pero no es recomendable).

```
sh scripts/train.sh P01_01
```

Pudes resumir el entrenamiento si en el script añades --ckpt_path ckpts/rel/$VID\/epoch=2.ckpt por ejemplo. Puedes usar cualquier ckpt que quieras

Para entrenar CLIP/DINO después de entrenar la geometría usar este script.

```
sh scripts/train_clip.sh P01_01
```

Hay que indicar con --ckpt_path_pretrained el ckpt que vas a usar del modelo de geometría para empezar a entrenar.

En ambos casos hay una variedad de argumentos que se pueden modificar (mirar opt.py) o preguntarme.

## Evaluation
Para evaluar el modelo puedes usar el siguiente script de estas maneras:

### Visualisations and metrics per scene

To evaluate the scene with Video ID `P01_01`, use the following command:

```
sh scripts/eval.sh rel P01_01 rel 'masks' 0 0
```
The las two arguments represent the number of frames to use in the evaluation. 0 means that we use every frame.

The results are saved in `results/rel`. The subfolders contain a txt file containing the mAP and PSNR scores per scene and visualisations per sample.

You can find all scene IDs in the EPIC-Diff data folder (e.g. `P01_01`, `P03_04`, ... `P21_01`).

### Rendering a video with separation of background, foreground and actor

To visualise the different model components of a reconstructed video (as seen on top of this page) from
1) the ground truth camera poses corresponding to the time of the video
2) and a fixed viewpoint,
use the following command:

```
sh scripts/eval.sh rel P01_01 rel 'summary' 0 0
```

This will result in a corresponding video in the folder `results/rel/P01_01/summary`.

The fixed viewpoints are pre-defined and correspond to the ones that we used in the videos provided in the supplementary material. You can adjust the viewpoints in `__init__.py` of `dataset`.

### Rendering a video with CLIP embeddings

Add in eval.sh the arg --positive_queries with the quueries you want to render (e.g. --positive_queries "human arms","cooking")

```
sh scripts/eval.sh rel P01_01 rel 'clip_video' 0 0
```
