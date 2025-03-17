import json

import numpy as np
import skimage.draw
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import torch
import os


class LabelLoader:
    def __init__(self, path_json, imhw_src, imhw_dst=[270, 480], each_nth=1, **kwargs):
        with open(path_json, "r") as f:
            data = json.load(f)

        self.data = data
        
        # resolution used for annotating not stored in VIA, defined here as 'src'
        self.imhw_src = imhw_src
        self.imhw_src_est = [0, 0]

        # rescale polygons to 'dst'
        self.imhw_dst = imhw_dst

        if len(self.data["attribute"]) == 0:
            self.labels = {}
        else:
            self.labels = {
                str(int(k) + 1): v for k, v in data["attribute"]["1"]["options"].items()
            }
        
        # via ID and filename
        self.vid2fn = {k: data["file"][k]["fname"] for k in data["file"]}
        self.fn2vid = {v: k for k, v in self.vid2fn.items()}

        self.vid_available = sorted(
            [*{*[data["metadata"][k]["vid"] for k in data["metadata"]]}]
        )
        self.fn_available = sorted(
            [*{*[self.vid2fn[vid] for vid in self.vid_available]}]
        )
        
        self.fn_available = self.fn_available[::each_nth]
        print('There are only ',len(self.fn_available), 'available frames for evaluating the model')

        # meta data ID
        self.mid2vid = {k: data["metadata"][k]["vid"] for k in data["metadata"]}
        self.vid2mid = {k: [] for k in self.vid2fn}
        for mid in self.mid2vid:
            vid = self.mid2vid[mid]
            self.vid2mid[vid].append(mid)

        # init src resolution
        for fn in self.fn_available:
            self.load_labels(fn)
            

        if self.imhw_src != [270, 480]:
            assert self.imhw_src[0] == self.imhw_src_est[0], [
                self.imhw_src[0],
                self.imhw_src_est[0],
            ]
            assert self.imhw_src[1] == self.imhw_src_est[1], [
                self.imhw_src[1],
                self.imhw_src_est[1],
            ]

    def load_polygon(self, mid, estimate_shape=[]):
        xy = self.data["metadata"][mid]["xy"][1:]
        xy = [(x, y) for x, y in zip(xy[1::2], xy[0::2])]
        pg = np.array(xy)
        self.imhw_src_est[0] = round(max(self.imhw_src_est[0], pg[:, 0].max()))
        self.imhw_src_est[1] = round(max(self.imhw_src_est[1], pg[:, 1].max()))
        return pg

    def draw_polygon(self, pg, im=None):
        if im is None:
            im = np.zeros(self.imhw_dst, "bool")
        rr, cc = skimage.draw.polygon(
            pg[:, 0] / self.imhw_src[0] * self.imhw_dst[0],
            pg[:, 1] / self.imhw_src[1] * self.imhw_dst[1],
            im.shape,
        )
        im[rr, cc] = 1
        return im

    def load_mask(self, fn):
        im_labeled = self.load_labels(fn)
        mask = im_labeled > 0
        mask = mask.astype("uint8")
        return mask

    def load_labels(self, fn):
        im_labeled = np.zeros(self.imhw_dst, dtype="uint8")
        try:
            vid = self.fn2vid[fn]
        except:
            vid = self.fn2vid[fn.replace("jpg", "bmp")]
        for mid in self.vid2mid[vid]:
            if len(self.labels) == 0:
                label = 1
            else:
                if len(self.data["metadata"][mid]["av"]) == 0:
                    label = 0
                else:
                    label = int(self.data["metadata"][mid]["av"]["1"])
            pg = self.load_polygon(mid)
            dpg = self.draw_polygon(pg)
            im_labeled[dpg] = label + 1
        return torch.from_numpy(im_labeled)



class Object_MaskLoader:
    """Loads masks for a dataset initialised with a video ID."""

    def __init__(self, dataset, is_debug=False):
        self.frames_dir = os.path.join(dataset.root, "frames")
        self.v_id = dataset.vid
        self.full_dataset_image_paths = dataset.image_paths
        self.rendered_img_h = 128
        self.rendered_img_w = 228
        
        loader_cfg = {'imhw_src': [270, 480], 'root': '/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data/object_annotations', 'each_nth': 1}
        split = "test"
        annotations_path = os.path.join(loader_cfg["root"], self.v_id, split[:2] + ".json")
        self.loader = LabelLoader(annotations_path, **loader_cfg, imhw_dst=loader_cfg["imhw_src"])
        self.labels_id = self.loader.labels
        
        self.fnames = self.loader.fn_available
        self.fnames_bmp = [f.replace('jpg', 'bmp') for f in self.fnames]
        self.image_paths = [self.frames_dir + '/' + f for f in self.fnames_bmp]

        print(f"ID of loaded scene: {dataset.vid}.")
        print(f"Number of annotations: {len(self.image_paths)}.")
        print('Loader labels ID',self.loader.labels)

    def __getitem__(self, frame_number):
        mask = self.loader.load_labels(str(frame_number))
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.rendered_img_h, self.rendered_img_w), mode='nearest').squeeze()
        
        # Obtener los índices únicos presentes en la máscara
        unique_keys_in_mask = torch.unique(mask).int().tolist()
        
        # Filtrar los text_queries para solo incluir aquellos cuyas claves están en la máscara
        text_queries = {key: value for key, value in self.labels_id.items() if int(key) in unique_keys_in_mask and value != 'undefined'}
        sample_id = next((k for k, v in self.full_dataset_image_paths.items() if v == frame_number.replace('jpg', 'bmp')), None)

        return mask, text_queries, sample_id

#loader_cfg = {'imhw_src': [270, 480], 'root': '/home/lmur/Feature_Fields/Language-Aware-NeuralDiff/data/object_annotations', 'each_nth': 1}               
#loader = make_loader(None, 'cli', loader_cfg=loader_cfg, vid='P01_01', split='test')
#labels = loader.load_labels('IMG_0000009603.jpg')


