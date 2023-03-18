## :one: requirement
* __document：__ requirements.txt

**cleaning data from rearview**
1. get unique images
1. combine unique images with annotations(4 datasets)
1. match image_id with the id in CVAT
3. rename category into six classes
4. split data into train and test(8:2)
#i didnt do it here

## :two: restore object name
| pedestrian-rider| car| truck-bus-train |
| -------- | -------- | -------- |
| human     | vehicle     | vehicle     |

| motorcycle-bicycle | traffic light | traffic sign |
| -------- | -------- | -------- |
| bike     |  traffic light    | traffic sign     |


## :three: embedding
purpose : to generate the 2D or 3D representation that is visualized
output : hoping to categorize images that are alike

![](https://i.imgur.com/ZnFEaTs.png)
###### note: embeddings for 4342 images

![](https://i.imgur.com/vzUmDWL.png)
###### note: embeddings for 3433 images


## :four: uniqueness
purpose : building a representation that relates the samples to each other, and analyze this representation to output uniqueness scores for each sample
output : populates a uniqueness field on each sample that contains the sample’s uniqueness score

![](https://i.imgur.com/f5h2aLp.jpg)
###### note: delete uniqueness lower then 0.08 by embeddings


## :five: brightness
purpose : to analyze images brightness
out : mean of brightness

**clarity**
purpose : to analyze images clartiy
output : variable of Laplacian

![](https://i.imgur.com/mDPn7L6.png)

*conclusion*:x-axis represent brightness, y-axis represent clarity.  By the two dimension plot we can tell most pictures aren't clarity, and the brightness of pictures lend in the middle of the plot.

![](https://i.imgur.com/9ohsvCA.png)
*note*:x-axis represent x-axis of embeddings, y-axis represent y-axis of embeddings, z-axis represent brightness.

![](https://i.imgur.com/5G7JVfs.png)
*note*:x-axis represent x-axis of embeddings, y-axis represent y-axis of embeddings, z-axis represent clarity.


## :six: scene&weather
purpose : to label images by running bdd100k tagging model, backbone by ResNet-18
out : scene tags and weather tags

![](https://i.imgur.com/HxV4FaS.png)
*note*:x-axis represent seven type of taggings, y-axis represent the amount of exch taggings. 

![](https://i.imgur.com/yql4X19.png)
*note*:x-axis represent seven type of taggings, y-axis represent the amount of exch taggings.
