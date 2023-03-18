from restore import load_data
from img_delete import img_delete
from new_json import new_json
from fiftyone_data import fiftyone
from claritybright import claritybright
from sceneweather import sceneweather

load_data('/SSD/yvonna/BSDdataset/168/annotations/instances_default.json')
img_delete("BSD", "/SSD/yvonna/BSDdataset/", "/SSD/yvonna/codes/rearview/rearviewpackage/restore.json")
new_json('/SSD/yvonna/codes/rearview/rearviewpackage/restore.json', '/SSD/yvonna/codes/rearview/rearviewpackage/img_delete.csv')
fiftyone("BSD_img3446", "/SSD/yvonna/BSDdataset/", "/SSD/yvonna/codes/rearview/rearviewpackage/img_delete.json")
claritybright('/SSD/yvonna/codes/rearview/rearviewpackage/img_delete.json', '/SSD/yvonna/codes/rearview/rearviewpackage/img_delete.csv')
sceneweather('/SSD/yvonna/codes/rearview/rearviewpackage/restore.json', '/SSD/yvonna/BSDdataset/scene_4326/preds.json', '/SSD/yvonna/BSDdataset/weather_4326/preds.json')
