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

def claritybright(file_path, img_delete):

    with open(file_path, newline='') as jsonfile:
        data = json.load(jsonfile)
    
    images_df = []
    for item in data['images']:
        images_df.append(item)
    images_df = pd.DataFrame(images_df, columns=['file_name'])

    images_df['add'] = list(['/SSD/yvonna/BSDdataset/']*len(images_df['file_name']))
    images_df['file'] = images_df['add']+images_df['file_name']

    array_of_bri = []
    for i in range(len(images_df['file'])):
        im = Image.open(images_df['file'][i]).convert('L')
        stat = ImageStat.Stat(im)
        stat_mean = stat.mean[0]
        array_of_bri.append(stat_mean)
    array_of_bri = pd.DataFrame(array_of_bri).rename(columns={0:'brightness'})

    array_of_img = [] 
    for i in range(len(images_df['file'])):
        img = cv2.imread(images_df['file'][i], cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.Laplacian(img, cv2.CV_16S).var()
        array_of_img.append(laplacian)
    array_of_img = pd.DataFrame(array_of_img).rename(columns={0:'clarity'})

    ananlyze = pd.DataFrame().join([images_df, array_of_bri, array_of_img], how="outer").drop(['add'], axis=1)

    plt.figure(figsize=(40, 40), dpi=100)
    plt.scatter(ananlyze['brightness'], ananlyze['clarity'], c='purple', s=100, label='brightness&clarity')
    plt.title('brightness&clarity', fontsize=15)
    plt.xlabel('brightness')
    plt.ylabel('clarity')
    # for i, label in enumerate(ananlyze['file_name']):
    #     plt.annotate(label, (ananlyze['brightness'][i], ananlyze['clarity'][i]))
    plt.show()

    img_cvs = pd.read_csv(img_delete)  

    # brightness&embedding
    x = img_cvs['0']
    y = img_cvs['1']
    z = ananlyze['brightness']

    # figure
    fig = plt.figure()
    ax = Axes3D(fig)

    # scatter plot
    ax.scatter(x, y, z, c ='red')
    plt.title('brightness&embedding', fontsize=15)
    plt.show()


    #clarity&embedding
    x = img_cvs['0']
    y = img_cvs['1']
    z = ananlyze['clarity']

    # figure
    fig = plt.figure()
    ax = Axes3D(fig)

    # scatter plot
    ax.scatter(x, y, z, c ='blue')
    plt.title('clarity&embedding', fontsize=15)
    plt.show()

