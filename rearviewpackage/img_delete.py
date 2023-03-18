from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import pandas as pd
import json
from numpy import arange
from tqdm import tqdm
import itertools
import os 
import fiftyone as fo
import cv2
import fiftyone.brain as fob
import random
from fiftyone import ViewField as F
import torch                                                                                                                              
import torchvision
from PIL import Image
from torchvision.transforms import functional as func
import dill as pickle
from PIL import ImageStat
from mpl_toolkits.mplot3d.axes3d import Axes3D

def img_delete(name, data_path, labels_path):
    name = name
    data_path = data_path
    labels_path = labels_path

    # Import dataset by explicitly providing paths to the source media and labels
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        name=name
    )

    dataset = fo.load_dataset('BSD')

    # Generate visualization for image
    results = fob.compute_visualization(dataset, method='tsne')

    # open a file, where you want to store the data
    file = open('embedding', 'wb')

    # dump information to that file
    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

    # close the file
    file.close()

    # Load
    with open('embedding', 'rb') as f:
        results_import = pickle.load(f)

    results_duplicate = results_import.serialize()

    points = []
    for item in results_duplicate['points']:
        points.append(item)
    points = pd.DataFrame(points)

    uni = []
    for i in range(len(points[0])):
        for j in range(i, len(points[0])):
            if (i!=j) & (abs(points[0][i] - points[0][j])<0.2) & (abs(points[1][i] - points[1][j])<0.2):
                uni.append(i)
    uni = pd.DataFrame(uni).drop_duplicates()

    # 3446 images left
    filter = points[~points.index.isin(list(uni[0]))]
    filter.to_csv(r'/SSD/yvonna/codes/rearview/rearviewpackage/img_delete.csv')
