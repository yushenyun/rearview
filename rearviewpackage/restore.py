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


def load_data(path):

    # load json data
    with open(path, newline='') as jsonfile:
        data = json.load(jsonfile)

    # trans image data to df
    images_df = []
    for item in data['images']:
        images_df.append(item)
    images_df = pd.DataFrame(images_df)

    # trans annotation data to df
    annotations_df = []
    for item in data['annotations']:
        annotations_df.append(item)
    annotations_df = pd.DataFrame(annotations_df)

    filter1 = (annotations_df['category_id'] != 10)&(annotations_df['category_id'] != 11)
    annotations_df = annotations_df.loc[filter1]

    # get unique image id
    img_dup = annotations_df["image_id"].drop_duplicates()

    # filter annotated image
    filter = images_df['id'].isin(list(img_dup))
    filtered_images_df = images_df.loc[filter].set_index(arange(1,len(img_dup)+1)).rename(columns={'id':'image_id'})
    
    #paste annotation and image together
    together = pd.merge(annotations_df, filtered_images_df, on='image_id')

    # trans categories data to df and delete front_road_seg & cone
    categories_df = []
    for item in data['categories']:
        categories_df.append(item)
    categories_df = pd.DataFrame(categories_df)
    categories_df['name_index'] = categories_df['name']
    categories_df = categories_df.set_index('name_index')
    categories_df = categories_df.drop(['front_road_seg', 'cone'], axis=0)

    #load licenses
    licenses_df = data['licenses']

    #load info
    info_df = data['info']

    #load cate
    categories_change = categories_df[['id', 'name', 'supercategory']]
    categories_jnos = categories_change.to_json(orient = 'records')
    categories_jnos = json.loads(categories_jnos)

    #trun image to json
    image_change = together[['image_id','width','height','file_name','license','flickr_url','coco_url','date_captured']].rename(columns={'image_id':'id'}).drop_duplicates()
    image_json = image_change.to_json(orient = 'records')
    image_json = json.loads(image_json)

    #trun annotation to json
    annotation_change = together[['id','image_id','category_id','segmentation','area','bbox','iscrowd','attributes']]
    annotation_jnos = annotation_change.to_json(orient = 'records')
    annotation_jnos = json.loads(annotation_jnos)
 
    #make a empety json
    jsontext = {'licenses':licenses_df, 'info':info_df, 'categories':categories_jnos, 'images':image_json, 'annotations':annotation_jnos}


    # Merge annotation for BDD categories
    def merge_BDD_annotations(json_obj):
        annos = json_obj["annotations"]

        new_categories = [
            {"supercategory": "human", "id": 1, "name": "pedestrian-rider"},
            {"supercategory": "vehicle", "id": 2, "name": "car"},
            {"supercategory": "vehicle", "id": 3, "name": "truck-bus-train"},
            {"supercategory": "bike", "id": 4, "name": "motorcycle-bicycle"},
            {"supercategory": "traffic light", "id": 5, "name": "traffic light"},
            {"supercategory": "traffic sign", "id": 6, "name": "traffic sign"}]
        json_obj["categories"] = new_categories

        def merge(label):
            return {
                1: 1,  # Pedestrian
                2: 1,  # rider
                3: 2,  # car
                4: 3,  # truck
                5: 3,  # bus
                6: 3,  # rail_car
                7: 4,  # motorcycle
                8: 4,  # bicycle
                9: 5,  # traffic light
                12: 6, # sign
                
            }[label]

        for anno in tqdm(annos):
            category = anno['category_id']
            anno['category_id'] = merge(category)

        return json_obj

    text = merge_BDD_annotations(jsontext)

    #dump
    out_file = open("restore.json", "w") 
    json.dump(text, out_file, indent=4)
    out_file.close()
