# Dynamic Image-Video Feature Fields (DIV-FF)

## ✨✨ Accepted in CVPR 2025!! ✨✨
[🎥 Ver video](P13_03-_online-video-cutter.com_.gif)

## Dataset
Descargar dataset de https://drive.google.com/file/d/1_FkZ1tXdW3JdTbEeiD-p4jOL1qvvKH46/view?usp=sharing y copiar la carpeta data al directorio donde tengas el repositorio descargado

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
