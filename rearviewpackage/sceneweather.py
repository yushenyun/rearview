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

def sceneweather(data, scene, weather):

    def restore(data):

        # load json data
        with open(data, newline='') as jsonfile:
            data = json.load(jsonfile)

        # trans image data to df
        images_df = []
        for item in data['images']:
            images_df.append(item)
        images_df = pd.DataFrame(images_df)

        #substr picture name out
        picture = []
        for i in range(len(images_df['file_name'])):
            pic = images_df['file_name'][i].split('/')
            picture.append(pic[-1])
        picture = pd.DataFrame(picture).rename(columns={0: 'name'})

        together = pd.merge(images_df, picture, left_index=True, right_index=True)

        name = together[['id', 'file_name', 'name']]
        
        return name


    def scene_4342(scene):

        # load json data
        with open(scene, newline='') as jsonfile:
            data = json.load(jsonfile)

        pred_class_df = []
        for item in data['pred_class']:
            pred_class_df.append(item)
        pred_class_df = pd.DataFrame(pred_class_df).rename(columns={0:'scene'})

        return pred_class_df


    def weather_4342(weather):

        # load json data
        with open(weather, newline='') as jsonfile:
            data = json.load(jsonfile)

        pred_class_df = []
        for item in data['pred_class']:
            pred_class_df.append(item)
        pred_class_df = pd.DataFrame(pred_class_df).rename(columns={0:'weather'})

        return pred_class_df

    together = pd.DataFrame().join([restore(data), scene_4342(scene)['scene'], weather_4342(weather)['weather']], how="outer")


    tunnel = 0
    residential = 0
    parking_lot = 0
    city_street = 0
    gas_stations = 0
    highway = 0
    undefine_scene = 0

    for i in range(len(together['scene'])):
        if 'tunnel' in str(together['scene'][i]):
            tunnel = tunnel + 1
        elif 'residential' in str(together['scene'][i]):
            residential = residential + 1
        elif 'parking lot' in str(together['scene'][i]):
            parking_lot = parking_lot + 1  
        elif 'city street' in str(together['scene'][i]):
            city_street = city_street + 1
        elif 'gas stations' in str(together['scene'][i]):
            gas_stations = gas_stations + 1    
        elif 'highway' in str(together['scene'][i]):
            highway = highway + 1      
        else:
            undefine_scene = undefine_scene + 1

    scene_dict = {
        'situation': ['tunnel', 'residential', 'parking lot', 'city street', 'gas stations', 'highway', 'undefined'],
        'count': [tunnel, residential, parking_lot, city_street, gas_stations, highway, undefine_scene]
    }    
    scene_df = pd.DataFrame(scene_dict)

    situation = ['tunnel', 'residential', 'parking lot', 'city street', 'gas stations', 'highway', 'undefined']
    count = [tunnel, residential, parking_lot, city_street, gas_stations, highway, undefine_scene]
    x = np.arange(len(situation))
    plt.bar(x, count, color=['darkslategrey', 'teal', 'lightseagreen', 'deepskyblue', 'aquamarine', 'cadetblue', 'steelblue'])
    plt.xticks(x, situation)
    plt.xlabel('situation')
    plt.ylabel('count')
    plt.title('scene')
    plt.show()



    rainy = 0
    snowy = 0
    clear = 0
    overcast = 0
    partly_cloudy = 0
    foggy = 0
    undefine_weather = 0

    for i in range(len(together['weather'])):
        if 'rainy' in str(together['weather'][i]):
            rainy = rainy + 1
        elif 'snowy' in str(together['weather'][i]):
            snowy = snowy + 1
        elif 'clear' in str(together['weather'][i]):
            clear = clear + 1  
        elif 'overcast' in str(together['weather'][i]):
            overcast = overcast + 1
        elif 'partly cloudy' in str(together['weather'][i]):
            partly_cloudy = partly_cloudy + 1    
        elif 'foggy' in str(together['weather'][i]):
            foggy = foggy + 1      
        else:
            undefine_weather = undefine_weather + 1
        
    weather_dict = {
        'situation': ['rainy', 'snowy', 'clear', 'overcast', 'partly cloudy', 'foggy', 'undefined'],
        'count': [rainy, snowy, clear, overcast, partly_cloudy, foggy, undefine_weather]
    }    
    weather_df = pd.DataFrame(weather_dict)

    situation = ['rainy', 'snowy', 'clear', 'overcast', 'partly cloudy', 'foggy', 'undefined']
    count = [rainy, snowy, clear, overcast, partly_cloudy, foggy, undefine_weather]
    x = np.arange(len(situation))
    plt.bar(x, count, color=['darkred', 'firebrick', 'brown', 'indianred', 'lightcoral', 'salmon', 'rosybrown'])
    plt.xticks(x, situation)
    plt.xlabel('situation')
    plt.ylabel('count')
    plt.title('weather')
    plt.show()
