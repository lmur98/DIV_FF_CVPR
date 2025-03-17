import json

import numpy as np
import skimage.draw
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import torch
import os
import pickle

class Affordances_MaskLoader:
    """Loads masks for a dataset initialised with a video ID."""

    def __init__(self, dataset, is_debug=False):
        self.frames_dir = os.path.join(dataset.root, "frames")
        self.v_id = dataset.vid
        self.full_dataset_image_paths = dataset.image_paths
        self.rendered_img_h = 128
        self.rendered_img_w = 228
        
        root = '/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data'
        annotations_path = os.path.join(root, 'affordance_annotations', self.v_id, 'aff_annotations.pkl')
        with open(annotations_path, 'rb') as f:
            self.loader = pickle.load(f)

        labels_id = self.loader['text_queries']
        labels_id = [label.replace('#C C ', '') for label in labels_id]#[label.split(' ')[2] for label in labels_id]#[label.replace('#C C ', '') for label in labels_id]
        self.labels_id = [label.replace('spagethi', 'pasta') for label in labels_id]
        self.fnames = self.loader['images'].keys()
        self.aff_masks = self.loader['images']

        print(f"ID of loaded scene: {dataset.vid}.")
        print(f"Number of annotations: {len(self.fnames)}.")
        print('Loader labels ID',self.labels_id)

    def __getitem__(self, frame_number):
        mask = self.aff_masks[frame_number] #(h, w, len(self.labels_id))
        text_queries = self.labels_id
        sample_id = next((k for k, v in self.full_dataset_image_paths.items() if v == frame_number.replace('jpg', 'bmp')), None)
        return mask, text_queries, sample_id