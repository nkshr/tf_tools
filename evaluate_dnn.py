import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import cv2
import csv
from lib import eval_info
from lib import util

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    einfo = eval_info.eval_info()
    einfo.init(flags.images, flags.labels, flags.num_images)
    
    graph = util.load_graph(flags.graph)

    with tf.Session(graph = graph) as sess:
        input_tensor = graph.get_tensor_by_name(flags.input_tensor_name)
        output_tensor = graph.get_tensor_by_name(flags.output_tensor_name)
        
        ground_truth_tensor = tf.placeholder(
            tf.float32,
            [None, einfo.get_class_count()],
            name = 'GroundTruthInput'
        )            
        
        if not flags.eval_classes:
            class_ids = range(einfo.get_class_count())
        else:
            class_ids = flags.eval_classes
            
        for class_id in class_ids:
            tf.logging.info('class {} is processed.'.format(class_id))
            cinfo= einfo.get_class_info(class_id)
                                
            for image_id in range(len(cinfo.iinfo_list)):
                iinfo = cinfo.iinfo_list[image_id]
                image_path = os.path.join(flags.image_dir, iinfo.name)
                if flags.debug:
                    tf.logging.info('Evaluate {} {}'.format(image_path, class_id))

                if flags.preprocess == 'rgb':
                    image = util.read_rgb_image(flags.input_width, flags.input_height, image_path)
                    if image is None:
                        tf.logging.error('Couldn\'t find {}'.format(image_path))
                    if flags.debug:
                        tf.logging.info("rgb image read")
                elif flags.preprocess == 'grayscale':
                    image = util.read_gray_image(flags.input_width, flags.input_height, image_path)
                    if image is None:
                        tf.logging.error('Couldn\'t find {}'.format(image_path))
                    if flags.debug:
                        tf.logging.info("gray image read")
                elif flags.preprocess == 'blur':
                    tf.logging.error('preprocess \'blur\' is not implemented yet.')
                    image = util.read_blur_image(flags.input_width, flags.input_height, image_path)
                    if image is None:
                        tf.logging.error('Couldn\'t find {}'.format(image_path))                        
                else:
                    tf.logging.error('preprocess \'{}\' is not allowed.'.format(flags.preprocess))
                   
                results = sess.run(output_tensor,
                                   feed_dict={
                                       input_tensor : image})

                results = np.squeeze(results)
                prob = results[class_id]
                iinfo.prob = prob
                
                sorted_indexes = results.argsort()[::-1]
                for i in range(len(sorted_indexes)):
                    if sorted_indexes[i] == class_id:
                        iinfo.rank = i
                        
                if flags.debug:
                    tf.logging.info('Result : {} {} {}'.format(image_path, iinfo.rank, iinfo.prob))

                for i in range(5):
                    tmp_cid = sorted_indexes[i]
                    prob = results[tmp_cid]
                    iinfo.top5[i]['class_id'] = tmp_cid
                    iinfo.top5[i]['prob'] = prob                    

    operations_path = os.path.join(flags.result_dir, 'operations.txt')
    with open(operations_path, 'w') as f:
        for op in graph.get_operations():
            f.write(op.name+'\n')

    einfo.take_statistcs()
    #einfo.sort_by_top1_rate()

    einfo.write(flags.result_dir)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph',
        type = str,
        default = 'data/flower/fine_tuned_graph.pb',
        help=''
    )
    parser.add_argument(
        '--input_tensor_name',
        type = str,
        default = 'input:0',
        help = ''
    )
    parser.add_argument(
        '--output_tensor_name',
        type = str,
        default = 'softmax:0',
        help = ''
    )
    parser.add_argument(
        '--image_dir',
        type = str,
        default = 'data/flower_photos',
        help = ''
    )
    parser.add_argument(
        '--images',
        type = str,
        default = 'data/flower/images.txt',
        help = ''
    )
    parser.add_argument(
        '--labels',
        type = str,
        default = 'data/flower/labels.txt',
        help = ''
    )
    parser.add_argument(
        '--debug',
        action = 'store_true',
    )
    parser.add_argument(
        '--input_width',
        default = 299
    )
    parser.add_argument(
        '--input_height',
        default = 299
    )
    parser.add_argument(
        '--eval_classes',
        nargs = '*',
        type = int,
        help = 'If eval_classes are not  specified, all classes are evaluated.Otherwise only specified classes are evaluated.'
    )
    parser.add_argument(
        '--preprocess',
        type = str,
        default = 'rgb',
        help = 'if preprocess is needed, choose \"grayscale\"or \"blur\".'
    )
    parser.add_argument(
        '--num_images',
        type = int,
        default = -1,
        help = 'Maximum number of images to be evaluated per class.\nIf not specified, all images are evaluated.'
    )
    parser.add_argument(
        '--result_dir',
        type = str,
        default = 'data/einfo'
    )
    
    flags, unparsed = parser.parse_known_args()

    main()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
