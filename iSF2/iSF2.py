import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import base64
import json
from requests import Request, Session
import os, csv, glob
import collections
from bs4 import BeautifulSoup
import re

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

%matplotlib inline

PATH_TO_FROZEN_GRAPH = 'model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'label.pbtxt'
NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap('label.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
IMAGE_SIZE = (12, 8)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def recognize_captcha(str_image_path):
    bin_captcha = open(str_image_path, 'rb').read()

    str_encode_file = base64.b64encode(bin_captcha).decode("utf-8")

    str_url = "https://vision.googleapis.com/v1/images:annotate?key="
    
    # change into you API key
    str_api_key = "change into you API key"

    str_headers = {'Content-Type': 'application/json'}

    str_json_data = {
        'requests': [
            {
                'image': {
                    'content': str_encode_file
                },
                'features': [
                    {
                        'type': "TEXT_DETECTION",
                        'maxResults': 20
                    }
                ],
                "imageContext": {
                    "languageHints": [
                        "en"
                    ]
                }
            }
        ]
    }

    print("begin request")
    obj_session = Session()
    obj_request = Request("POST",
                          str_url + str_api_key,
                          data=json.dumps(str_json_data),
                          headers=str_headers
                          )
    obj_prepped = obj_session.prepare_request(obj_request)
    obj_response = obj_session.send(obj_prepped,
                                    verify=True,
                                    timeout=60
                                    )
    print("end request")

    if obj_response.status_code == 200:
        with open('data.json', 'w') as outfile:
            json.dump(obj_response.text, outfile)
        return obj_response.text
    else:
        return "error"

# change into your image path
image_path = 'change into your image'
image = Image.open(image_path)
cv2_image = cv2.imread(image_path)
image_np = load_image_into_numpy_array(image)
image_np_expanded = np.expand_dims(image_np, axis=0)
output_dict = run_inference_for_single_image(image_np, detection_graph)
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)
fig = plt.figure(figsize=IMAGE_SIZE)
ax = fig.gca()
ax.grid(False)
plt.imshow(image_np)
alpha = 0.5
pil_mask = Image.fromarray(np.uint8(255.0*alpha*output_dict.get('detection_masks')[1])).convert('L')
im_height, im_width, im_color = cv2_image.shape
Laxis_xmin = int(output_dict['detection_boxes'][0][0]*im_height)
Laxis_ymin = int(output_dict['detection_boxes'][0][1]*im_width)
Laxis_xmax = int(output_dict['detection_boxes'][0][2]*im_height)
Laxis_ymax = int(output_dict['detection_boxes'][0][3]*im_width)
curve_xmin = int(output_dict['detection_boxes'][1][0]*im_height)
curve_ymin = int(output_dict['detection_boxes'][1][1]*im_width)
curve_xmax = int(output_dict['detection_boxes'][1][2]*im_height)
curve_ymax = int(output_dict['detection_boxes'][1][3]*im_width)
ymin = int(output_dict['detection_boxes'][1][0]*im_height)
ymax = int(output_dict['detection_boxes'][0][2]*im_height)
xmin = int(output_dict['detection_boxes'][0][1]*im_width)
xmax = int(output_dict['detection_boxes'][1][3]*im_width)
print("xmin={}, xmax={}, ymin={}, ymax{}".format(xmin, xmax, ymin, ymax))
log_scale = cv2_image[0:im_height, 50:xmin]
save_log_scale = IMAGE_OUT + '/' + 'log_scale' + '.png'
cv2.imwrite(save_log_scale, log_scale)
aa = ymax + 50
ac = xmax + 20  
data_x = json.loads(recognize_captcha(save_x_scale))
data_x = data_x["responses"]
new_list_x = []
for i in data_x:
  try:
    new_list_x.append(i["fullTextAnnotation"]["text"])
  except:
    pass
  true_data_x = ",".join(new_list_x)
  kubetsunashi_x = true_data_x.lower()
  kubetsunashi_x = kubetsunashi_x.replace(" ", "")
  print(kubetsunashi_x)
  kubetsunashi_x = kubetsunashi_x.splitlines()
  l = []
  for t in kubetsunashi_x:
    try:
        l.append(float(t))
    except ValueError:
        pass
  x_memori = float(max(l))
  print("x_memori={}".format(x_memori))

data_log = json.loads(recognize_captcha(save_log_scale))
data_log = data_log["responses"]
new_list_log = []
for i in data_log:
  try:
    new_list_log.append(i["fullTextAnnotation"]["text"])
  except:
    pass
  true_data_log = ",".join(new_list_log)
  kubetsunashi_log = true_data_log.lower()
  kubetsunashi_log = kubetsunashi_log.replace(" ", "")
  count10 = kubetsunashi_log.count('10')
  print(kubetsunashi_log)
  kubetsunashi_log = kubetsunashi_log.splitlines()
  y_memori = len(kubetsunashi_log)
  print("count10={}".format(count10))
  print("y_memori={}".format(y_memori))

x_length = xmax - xmin
x_length = x_length / x_memori
x_for_2Gy = int(xmin + 2*x_length)
print("x_for_2Gy={}".format(x_for_2Gy))
imgArray = np.asarray(pil_mask)
curve_pixel_indexes = np.where(imgArray[:, x_for_2Gy] == 127)
print("curve_pixel_indexes={}".format(curve_pixel_indexes))
ytarget = np.min(curve_pixel_indexes)
y_length = ymax - ymin
print("ytarget={}, y_length={}".format(ytarget, y_length))
cc = (y_memori -2) / (y_memori-1) * y_length
print(cc)
dd = ymax - ytarget - cc
ee = y_length / (y_memori - 1)
ff = dd / ee
sf2 = 10**ff *10
print("sf2={}".format(sf2))