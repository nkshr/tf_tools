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
    einfo = eval_info.eval_info()
    einfo.init(flags.images, flags.labels, flags.num_images)
    
    graph = util.load_graph(flags.graph)

    with tf.Session(graph = graph) as sess:
        input_tensor = graph.get_tensor_by_name(flags.input_tensor_name)
        output_tensor = graph.get_tensor_by_name(flags.output_tensor_name)

        if flags.preprocess == 'grayscale':
            jpeg_data_tensor, decoded_data_tensor = util.add_jpeg_grayscale_decoding(
                flags.input_width, flags.input_height,
            )
            print('grayscale conversion was choosed as preprocess.')
        else:
            jpeg_data_tensor, decoded_data_tensor = util.add_jpeg_decoding(
                flags.input_width, flags.input_height,
                flags.input_depth
            )
            print('No preprocess was choosed.')
        
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
            print('class', class_id, 'is processed.')
            cinfo= einfo.get_class_info(class_id)
                                
            for image_id in range(len(cinfo.iinfo_list)):
                iinfo = cinfo.iinfo_list[image_id]
                image_path = os.path.join(flags.image_dir, iinfo.name)
                if flags.debug:
                    print('Evaluate', image_path, class_id)

                if not gfile.Exists(image_path):
                    print(image_path, "doesn't exist.")
                    
                jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()
                if flags.debug:
                    raw_data = sess.run(raw_data_tensor,
                             feed_dict={jpeg_data_tensor : jpeg_data})
                    if flags.preprocess == 'grayscale':
                        cv2.imwrite('test.png', raw_data)
                    else:
                        cv2.imwrite('test.png', cv2.cvtColor(raw_data, cv2.COLOR_RGB2BGR))
                        
                resized_data = sess.run(decoded_data_tensor,
                                       feed_dict={jpeg_data_tensor : jpeg_data})
                            
                results = sess.run(output_tensor,
                                   feed_dict={
                                       input_tensor : resized_data})

                results = np.squeeze(results)
                prob = results[class_id]
                iinfo.prob = prob
                
                sorted_indexes = results.argsort()[::-1]
                for i in range(len(sorted_indexes)):
                    if sorted_indexes[i] == class_id:
                        iinfo.rank = i
                        
                if flags.debug:
                    print('Result :', image_path, iinfo.rank, iinfo.prob)

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
        default = 'data/flower_graph.pb',
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
        default = 'data/flower_images.txt',
        help = ''
    )
    parser.add_argument(
        '--labels',
        type = str,
        default = 'data/flower_labels.txt',
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
        '--input_depth',
        type = int, 
        default = 3
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
        default = None,
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
