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

def fiftyone(name, data_path, labels_path):
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

    dataset = fo.load_dataset('BSD_img3446')

    session = fo.launch_app(dataset)

    results = fob.compute_visualization(dataset)

    # Generate scatterplot
    plot = results.visualize()
    plot.show(height=800)
    session.plots.attach(plot)

    fob.compute_uniqueness(dataset)

    # Sort by uniqueness (most unique first)
    dups_view = dataset.sort_by("uniqueness", reverse=True)

    # Open view in the App
    session.view = dups_view

