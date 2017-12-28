import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile
import cv2
import numpy as np
from six.moves import urllib
import tensorflow as tf
import math

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size, learning_rate):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    bottleneck_tensor_size: How many entries in the bottleneck vector.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction

def add_jpeg_decoding(input_width, input_height, input_depth):
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  image_as_float = tf.cast(decoded_image, dtype = tf.float32)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype = tf.int32)
  image_4d = tf.expand_dims(image_as_float, 0)
  resized_image = tf.image.resize_bilinear(image_4d,
                                           resize_shape_as_int)
  mean = tf.reduce_mean(resized_image)
  squared_diffs = tf.squared_difference(resized_image, mean)
  std = tf.sqrt(tf.reduce_mean(squared_diffs)) 
  offset_image = tf.subtract(resized_image, mean)
  mul_image = tf.multiply(offset_image, 1.0 / std)
  return jpeg_data, mul_image

def add_jpeg_grayscale_decoding(input_width, input_height):
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=1)
  image_3c = tf.image.grayscale_to_rgb(decoded_image)
  image_as_float = tf.cast(image_3c, dtype=tf.float32)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  image_4d = tf.expand_dims(image_as_float, 0)
  resized_image = tf.image.resize_bilinear(image_4d,
                                           resize_shape_as_int)
  mean = tf.reduce_mean(resized_image)
  squared_diffs = tf.squared_difference(resized_image, mean)
  std = tf.sqrt(tf.reduce_mean(squared_diffs)) 
  offset_image = tf.subtract(resized_image, mean)
  mul_image = tf.multiply(offset_image, 1.0 / std)

  return jpeg_data, mul_image

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name="")

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(image_reader)

  return result

def create_image_lists(fname, test_rate, val_rate, num_classes):
    image_lists = [{'train' : list(), 'test' : list(), 'val' : list() } for i in range(num_classes)]
    random.seed(1)
    with open(fname, 'r') as f:
        for line in f.readlines():
            toks = line.split()
            name = toks[0]
            class_id = int(toks[1])
            # if class_id not in image_lists:
            #     image_lists[class_id]={
            #         'train' : list(),
            #         'test' : list(),
            #         'val' : list()
            #     }

            rval = random.randrange(100) * 0.01

            if rval < val_rate:
                image_lists[class_id]['val'].append(name)
            elif rval < val_rate + test_rate:
                image_lists[class_id]['test'].append(name)
            else:
                image_lists[class_id]['train'].append(name)
            
    return image_lists

def write_out_image_lists(image_lists, ftrain, ftest, fval):
    flist = {'train' : ftrain, 'test' : ftest, 'val' : fval}
    for dtype, fname in flist.items():
        with open(fname, 'w') as f:
            for class_id in range(len(image_lists)):
                for image in image_lists[class_id][dtype]:
                    f.write(image+' '+ str(class_id)+'\n')

# def sort_image_lists_by_class_id(image_lists):
#     class_ids = []
#     for k, v in image_lists.items():
#         class_ids.append(int(k))

#     sorted_indexes = np.argsort(class_ids)
#     sorted_image_lists = {}
#     for index in sorted_indexes:
#         class_id = class_ids[index]
#         sorted_image_lists[class_id] = image_lists[str(class_id)]

#     #for k in sorted_image_lists.keys():
#     #  print(k)
      
#     return sorted_image_lists



def get_image_names(batch_size, itype, image_lists):
  class_count = len(image_lists)
  images = []
  ground_truths = []
  if batch_size < 0:
    for class_id in range(class_count):
      for image in image_lists[class_id][itype]:
        images.append(image)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[class_id] = 1.0
        ground_truths.append(ground_truth)
  else:
    for i in range(batch_size):
      while True:
        class_id = random.randrange(class_count)
        num_images = len(image_lists[class_id][itype])
        if num_images > 0:
          break
        elif num_images == 0:
          print(class_id, 'is empty.')

      image_index = random.randrange(num_images)
      image = image_lists[class_id][itype][image_index]
      images.append(image)
      ground_truth = np.zeros(class_count, dtype = np.float32)
      ground_truth[class_id] = 1.0
      ground_truths.append(ground_truth)

  return images, ground_truths

def get_bottlenecks(itype, preprocess, batch_size, width, height,
                     image_dir, image_lists,
                     bottleneck_tensor, input_tensor, sess):
  image_names, ground_truths = get_image_names(batch_size, itype, image_lists)
  bottlenecks = []
  if preprocess == 'rgb':
    for image_name in image_names:
      image_path = os.path.join(image_dir, image_name)
      image = read_rgb_image(width, height, image_path)
      bottleneck = sess.run(bottleneck_tensor, feed_dict = {input_tensor : image})
      bottleneck = np.squeeze(bottleneck)
      bottlenecks.append(bottleneck)
  elif preprocess == 'gray':
    for image_name in image_names:
      image_path = os.path.join(image_dir, image_name)
      image = read_gray_image(width, height, image_path)
      bottleneck = sess.run(bottleneck_tensor, feed_dict = {input_tensor : image})
      bottleneck = np.squeeze(bottleneck)
      bottlenecks.append(bottleneck)
  elif preprocess == 'blur':
    #tf.logging.error('preprocess \"blur\" is not implemented yet in get_bottlenecks.')
    for image_name in image_names:
      image_path = os.path.join(image_dir, image_name)
      image = read_blur_image(width, height, image_path)
      bottleneck = sess.run(bottleneck_tensor, feed_dict = {input_tensor : image})
      bottleneck = np.squeeze(bottleneck)
      bottlenecks.append(bottleneck)
  else:
    tf.logging.error('preprocess \"{}\" is not allowed.'.format(preprocess))
    exit(1)

  return bottlenecks, ground_truths

def read_preprocessed_image(width, height, gray_rate, blur_rate, image_path):
    rval = random.randrange(10) * 0.1
    if rval < gray_rate:
      #tf.logging.info('gray')
      image = read_gray_image(width, height, image_path)
    elif gray_rate <= rval and rval < gray_rate + blur_rate:
      #tf.logging.info('blur')
      image = read_blur_image(width, height, image_path)
    else:
      #tf.logging.info('rgb')
      image = read_rgb_image(width, height, image_path)
    return image
  
def get_expanded_bottlenecks(itype,
                             gray_rate, blur_rate,
                             batch_size, width, height,
                             image_dir, image_lists,
                             bottleneck_tensor, input_tensor, sess):
  image_names, ground_truths = get_image_names(batch_size, itype, image_lists)
  bottlenecks = []
  for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    image = read_preprocessed_image(width, height, gray_rate, blur_rate, image_path)
    bottleneck = sess.run(bottleneck_tensor, feed_dict = {input_tensor :image})
    bottleneck = np.squeeze(bottleneck)
    bottlenecks.append(bottleneck)

  return bottlenecks, ground_truths


def read_gray_images(image_dir, image_names):
  images = []
  for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
      tf.logging.error('File does not exist {}'.format(image_path))
    images.append(image)
  return images

def read_rgb_images(image_dir, image_names):
  images = []
  for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
      tf.logging.error('File does not exist {}'.format(image_path))
    images.append(image)
  return images

def save_graph_to_file(sess, graph, graph_file_name, final_tensor_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [final_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def normalize(v):
  sum = 0
  for i in range(len(v)):
    sum += v[i]

  for i in range(len(v)):
    v[i] /= sum
  return v

def remove_invalid_labels(results, labels, invalid_labels):
  invalid_idxs = list()
  for invalid_label in invalid_labels:
    try:
      idx = labels.index(invalid_label)
      labels.pop(idx)
      results = np.delete(results, idx)
      print(invalid_label, "is removed.")
    except ValueError as e:
      print("labels doesn't contain", invalid_label, ".")

  results = normalize(results)

  return results, labels

def normalize_image(img):
  #We should normalize image by following commentout-region, but We use mean=0 and std=255, because Google fixs mean and std.
  # mean = np.mean(img, dtype=np.float32)
  # std = np.std(img, dtype=np.float32)
  # if std < sys.float_info.epsilon:
  #   tf.logging.info("std is too small.")
  #   std = 1.0
  mean = 0
  std = 255
  nimg = (img - mean) / std
  return nimg

def read_rgb_image(width, height, image_path):
  bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  if bgr_image is None:
    tf.logging.error('{} does not exist.'.format(image_path))
    return None
  rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
  image_as_float = rgb_image.astype(np.float32)
  resized_image = cv2.resize(image_as_float, (width, height))
  normalized_image = normalize_image(resized_image)
  expanded_image = np.expand_dims(normalized_image, 0)
  return expanded_image

def read_gray_image(width, height, image_path):
  gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if gray_image is None:
    tf.logging.error('{} does not exist.'.format(image_path))
    return None
  image_as_3c = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
  image_as_float = image_as_3c.astype(np.float32)
  resized_image = cv2.resize(image_as_float, (width, height))
  normalized_image = normalize_image(resized_image)
  expanded_image = np.expand_dims(normalized_image, 0)
  return expanded_image

def read_blur_image(width, height, image_path):
  bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
  if bgr_image is None:
    tf.logging.error('{} does not exist.'.format(image_path))
    return None
  rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
  image_as_float = rgb_image.astype(np.float32)
  ksize = 10 + random.randrange(90)
  if ksize % 2 == 0:
    ksize += 1
  sigma = 10 + random.randrange(90)
  blur_image = cv2.GaussianBlur(image_as_float, (ksize, ksize), sigma)
  resized_image = cv2.resize(blur_image, (width, height))
  normalized_image = normalize_image(resized_image)
  expanded_image = np.expand_dims(normalized_image, 0)
  return expanded_image
