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

from lib import util

if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  parser.add_argument("--image",
                       default="data/sunflower.jpg",
                       help="image to be classified (default data/sunflower.jpg)")
  parser.add_argument("--graph",
                      default="data/fine_tuned_graph.pb",
                      help="graph for image classification (default data/fine_tuned_graph.pb)")
  parser.add_argument("--labels",
                      default="data/flower_labels.txt",
                      help="file containing labels (default ./data/flower_lalbels.txt)")
  parser.add_argument("--input_height",
                      default=299,
                      type=int,
                      help="input height (default 299)")
  parser.add_argument("--input_width",
                      default=299,
                      type=int,
                      help="input width (default 299)")
  parser.add_argument("--input_depth",
                      default=3,
                      type=int,
                      help="input_depth (default 3)")
  parser.add_argument("--input_layer",
                      default="input",
                      help="name of input layer (default input)")
  parser.add_argument("--output_layer",
                      default="soft_max",
                      help="name of outpu layer (default soft_max")  
  parser.add_argument("--result",
                      default="data/result.txt",
                      help="name of result file (default data/result.txt")
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

  for op in graph.get_operations():
    print(op.name)
    
  #input_name = flags.input_layer
  #output_name = flags.output_layer
  #input_operation = graph.get_operation_by_name(input_name);
  #output_operation = graph.get_operation_by_name(output_name);
  input_tensor = graph.get_tensor_by_name(flags.input_tensor_name)
  output_tensor = graph.get_tensor_by_name(flags.output_tensor_name)
  
  with tf.Session(graph=graph) as sess:
    jpeg_data_tensor, decoded_data_tensor = util.add_jpeg_decoding(flags.input_width, flags.input_height, flags.input_depth)

    if not tf.gfile.Exists(flags.image):
      print('File does not exist {}'.format(flags.image))
    jpeg_data = tf.gfile.FastGFile(flags.image, 'rb').read()

    decoded_data = sess.run(decoded_data_tensor,
                            {jpeg_data_tensor : jpeg_data})

    #results = sess.run(output_operation.outputs[0],
    #                  {input_operation.outputs[0]: decoded_data})
    results = sess.run(output_tensor,
                       feed_dict = {input_tensor : decoded_tensor})
    
  results = np.squeeze(results)
  labels = util.load_labels(flags.labels)
  
  results, labels = util.remove_invalid_labels(results, labels, flags.invalid_labels)
  top_k = results.argsort()[:][::-1]
  
  with open(flags.result, "w") as f:
    for i in top_k:
      try:
        f.write(labels[i] + " " + str(results[i]) + "\n")
        if flags.debug:
          print(labels[i] + " " + str(results[i]))
          
      except IndexError as e:
        print(e, i)
        