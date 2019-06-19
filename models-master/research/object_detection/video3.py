import tensorflow as tf
import os
import sys
#We used utils folder that is including many tools.
from utils import label_map_util

# Object detection imports. There is a counting model
from api import object_counting_api

#Detcted .avi file
input_video = "test_1.mp4"
PATH_TO_CKPT = 'My_model_31_22/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

sys.path.append("..")

#Set of tools and Graph
detection_graph = tf.Graph()# We are using our model that is named 'My_model_31_22'
with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

fps = 24 # change it with your input video fps
width = 854 # change it with your input video width
height = 480 # change it with your input vide height
is_color_recognition_enabled = 0

object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects