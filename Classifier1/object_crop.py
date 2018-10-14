import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module
MODEL_NAME = 'inference_graph'
data_dir_path = u"./photo/"
IMAGE_OUT = "./photo_crop/"
file_list = os.listdir(r'./photo/')

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')
NUM_CLASSES = 3
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Path to image
for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    if ext == u'.png' or u'.jpeg' or u'.jpg':
        abs_name = data_dir_path + '/' + file_name
        image = cv2.imread(abs_name)
        image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)

    im_height, im_width, im_color = image.shape
    for i, box in enumerate(np.squeeze(boxes)):
        if(np.squeeze(scores)[i] > 0.8):
            if(np.squeeze(classes)[i] == 1):
                print("{}, ymin={}, xmin={}, ymax={}, xmax{}".format(np.squeeze(classes)[i], box[0]*im_height,box[1]*im_width,box[2]*im_height,box[3]*im_width))
                print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                Xmin = int(box[0]*im_height)
                Xmax = int(box[2]*im_height)
                Ymin = int(box[1]*im_width)
                Ymax = int(box[3]*im_width)
                new_image = cv2.imread(abs_name)
                roi = new_image[Xmin:Xmax, Ymin:Ymax]
                save_path = IMAGE_OUT + '/' + file_name + 'out_(' + str(i) +')' + '.jpeg'
                cv2.imwrite(save_path, roi)
            else: pass

cv2.destroyAllWindows()
