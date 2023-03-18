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

def new_json(path, img_delete):

    # load json data
    with open(path, newline='') as jsonfile:
        data = json.load(jsonfile)

    # trans image data to df
    images_df = []
    for item in data['images']:
        images_df.append(item)
    images_df = pd.DataFrame(images_df).rename(columns={'id':'image_id'})

    #read cvs
    img = pd.read_csv(img_delete)

    filter_img = images_df.index.isin(list(img['Unnamed: 0']))
    filtered_df = images_df.loc[filter_img]

    # trans annotation data to df
    annotations_df = []
    for item in data['annotations']:
        annotations_df.append(item)
    annotations_df = pd.DataFrame(annotations_df)

    #paste annotation and image together
    together = pd.merge(annotations_df, filtered_df, on='image_id')

    #load licenses
    licenses_df = data['licenses']

    #load info
    info_df = data['info']

    #load cate
    categories_df = data['categories']

    #trun image to json
    image_change = together[['image_id','width','height','file_name','license','flickr_url','coco_url','date_captured']].rename(columns={'image_id':'id'}).drop_duplicates()
    image_json = image_change.to_json(orient = 'records')
    image_json = json.loads(image_json)

    #trun annotation to json
    annotation_change = together[['id','image_id','category_id','segmentation','area','bbox','iscrowd','attributes']]
    annotation_jnos = annotation_change.to_json(orient = 'records')
    annotation_jnos = json.loads(annotation_jnos)
 
    #make a empety json
    jsontext = {'licenses':licenses_df, 'info':info_df, 'categories':categories_df, 'images':image_json, 'annotations':annotation_jnos}

    #dump
    out_file = open("img_delete.json", "w") 
    json.dump(jsontext, out_file, indent=4)
    out_file.close()
