# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import sys

import numpy as np
import tensorflow as tf

import copy
import math

import cv2

from lib import util

if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  parser.add_argument("--image",
                       default="data/flower/sunflower.jpg",
                       help="image to be classified (default data/flower/sunflower.jpg)")
  parser.add_argument("--graph",
                      default="data/flower/fine_tuned_graph.pb",
                      help="graph for image classification (default data/flower/fine_tuned_graph.pb)")
  parser.add_argument("--labels",
                      default="data/flower/labels.txt",
                      help="file containing labels (default ./data/flower/lalbels.txt)")
  parser.add_argument("--input_height",
                      default=299,
                      type=int,
                      help="input height (default 299)")
  parser.add_argument("--input_width",
                      default=299,
                      type=int,
                      help="input width (default 299)")
  parser.add_argument("--input_layer",
                      default="input",
                      help="name of input layer (default input)")
  parser.add_argument("--output_layer",
                      default="soft_max",
                      help="name of outpu layer (default soft_max")  
  parser.add_argument("--result",
                      default="data/result.txt",
                      help="name of result file (default data/flower/result.txt)")
  parser.add_argument("--invalid_labels",
                      default = [],
                      nargs="*",
                      help="labeles to be removed from result (default None)")
  parser.add_argument("--debug",
                      action = "store_true")
  parser.add_argument("--input_tensor_name",
                      type = str,
                      default = "input:0",
                      help="input tensor name (default input:0)")
  parser.add_argument("--output_tensor_name",
                      type = str,
                      default = "softmax:0",
                      help="output tensor name (default softmax:0)")
  flags = parser.parse_args()
    
  graph = util.load_graph(flags.graph)

  input_tensor = graph.get_tensor_by_name(flags.input_tensor_name)
  output_tensor = graph.get_tensor_by_name(flags.output_tensor_name)
  with tf.Session(graph=graph) as sess:
    if not tf.gfile.Exists(flags.image):
      print('File does not exist {}'.format(flags.image))

    bgr_img = cv2.imread(flags.image, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_as_float = rgb_img.astype(np.float32)
    resized_img = cv2.resize(img_as_float, (flags.input_width, flags.input_height),fx=0, fy=0,interpolation=cv2.INTER_LINEAR)
    normalized_img = util.normalize_image(resized_img)
    expanded_img = np.expand_dims(normalized_img, 0)
      
    results = sess.run(output_tensor,
                       feed_dict = {input_tensor : expanded_img})
  results = np.squeeze(results)
  labels = util.load_labels(flags.labels)
  
  results, labels = util.remove_invalid_labels(results, labels, flags.invalid_labels)
  top_k = results.argsort()[:][::-1]
  
  with open(flags.result, "w") as f:
    for i in top_k:
      try:
        f.write(labels[i] + " " + str(results[i]) + "\n")
          
      except IndexError as e:
        print(e, i)
        
    for i in top_k[0:3]:
      if flags.debug:
        print(labels[i] + " " + str(results[i]))
