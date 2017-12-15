import random
import argparse
import numpy as np

from lib import util

from datetime import datetime

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph',
        type = str,
        default = 'data/inception_v3_2016_08_28_frozen.pb'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/flower_photos',
    )
    parser.add_argument(
        '--images',
        type=str,
        default='data/flower_images.txt'
    )
    parser.add_argument(
        '--test_rate',
        type = int,
        default = 0.1
    )
    parser.add_argument(
        '--val_rate',
        type = int,
        default = 0.1
    )
    parser.add_argument(
        '--val_images',
        type=str,
        default='data/val_images.txt'
    )
    parser.add_argument(
        '--test_images',
        type=str,
        default='data/test_images.txt'
    )
    parser.add_argument(
        '--train_images',
        type=str,
        default='data/train_images.txt'
    )
    parser.add_argument(
        '--num_train_steps',
        type=int,
        default=500,
    )
    parser.add_argument(
        '--input_width',
        type=int,
        default=299
    )
    parser.add_argument(
        '--input_height',
        type=int,
        default=299
    )
    parser.add_argument(
        '--input_depth',
        type=int,
        default=3
    )
    parser.add_argument(
        '--resized_input_tensor_name',
        type=str,
        default='input:0',
    )
    parser.add_argument(
        '--bottleneck_tensor_name',
        type = str,
        default = 'InceptionV3/Logits/SpatialSqueeze:0'
    )
    parser.add_argument(
        '--operations',
        type = str,
        default = 'data/operations.txt'
    )
    parser.add_argument(
        '--bottleneck_tensor_size',
        type = str,
        default = 1001
    )
    parser.add_argument(
        '--preprocess',
        type = str,
        default = ''
    )
    parser.add_argument(
        '--final_tensor_name',
        type = str,
        default = 'softmax'
    )
    parser.add_argument(
        '--learning_rate',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--eval_step_interval',
        type = int,
        default = 250
    )
    parser.add_argument(
        '--summary_dir',
        type = str,
        default = 'data/retrain_logs',
    )
    parser.add_argument(
        '--intermediate_graph_dir',
        type = str,
        default = 'data/intermediate_graphs'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type = int,
        default = 250
    )
    parser.add_argument(
        '--fine_tuned_graph',
        type = str,
        default = 'data/flower_graph.pb'
    )
    parser.add_argument(
        '--labels',
        type = str,
        default = 'data/labels.txt'
    )
    parser.add_argument(
        '--train_batch_size',
        type = int,
        default = 100
    )
    parser.add_argument(
        '--val_batch_size',
        type = int,
        default = 100
    )
    parser.add_argument(
        '--test_batch_size',
        type = int,
        default = -1
    )
    flags, unparsed = parser.parse_known_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    if tf.gfile.Exists(flags.summary_dir):
        tf.gfile.DeleteRecursively(flags.summary_dir)
    tf.gfile.MakeDirs(flags.summary_dir)

    if (not tf.gfile.Exists(flags.intermediate_graph_dir)) and (flags.intermediate_store_frequency > 0):
        tf.gfile.MakeDirs(flags.intermediate_graph_dir)
        
    graph = util.load_graph(flags.graph)
    image_lists = util.create_image_lists(flags.images, flags.test_rate, flags.val_rate)    
    image_lists = util.sort_image_lists_by_class_id(image_lists)
    util.write_out_image_lists(image_lists, flags.train_images, flags.test_images, flags.val_images)

    with tf.Session(graph=graph) as sess:
        resized_input_tensor = graph.get_tensor_by_name(flags.resized_input_tensor_name)
        bottleneck_tensor = graph.get_tensor_by_name(flags.bottleneck_tensor_name)
        
        if flags.preprocess == "grayscale":
            jpeg_data_tensor, decoded_image_tensor = util.add_jpeg_grayscale_decoding(flags.input_width, flags.input_height)
            tf.logging.info('gracyscale conversion is applied.')
        else:
            jpeg_data_tensor, decoded_image_tensor = util.add_jpeg_decoding(
                flags.input_width, flags.input_height, flags.input_depth)

        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
        final_tensor) = util.add_final_training_ops(
            len(image_lists.keys()), flags.final_tensor_name, bottleneck_tensor,
            flags.bottleneck_tensor_size, flags.learning_rate)

        eval_step, pred = util.add_evaluation_step(
            final_tensor, ground_truth_input)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(flags.summary_dir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(flags.summary_dir + '/val', sess.graph)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(flags.num_train_steps):
            train_bottlenecks, train_ground_truths = util.get_image_batch(flags.train_batch_size,
                                                                          'train',
                                                                          flags.image_dir,
                                                                          image_lists,
                                                                          decoded_image_tensor,
                                                                          jpeg_data_tensor,
                                                                          bottleneck_tensor,
                                                                          resized_input_tensor,
                                                                          sess)        
            train_summary, _ = sess.run([merged, train_step], feed_dict = {bottleneck_input :  train_bottlenecks,
                                 ground_truth_input : train_ground_truths})
            train_writer.add_summary(train_summary, i)
            
            is_last_step = (i + 1 == flags.num_train_steps)
            if (i % flags.eval_step_interval) == 0 or is_last_step:
                train_bottlenecks, train_ground_truths = util.get_image_batch(flags.train_batch_size,
                                                                              'train',
                                                                              flags.image_dir,
                                                                              image_lists,
                                                                              decoded_image_tensor,
                                                                              jpeg_data_tensor,
                                                                              bottleneck_tensor,
                                                                              resized_input_tensor,
                                                                              sess)
                train_accuracy, cross_entropy_value = sess.run(
                    [eval_step, cross_entropy],
                    feed_dict = {bottleneck_input : train_bottlenecks,
                                 ground_truth_input : train_ground_truths})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entroy = %f' %
                                (datetime.now(), i, cross_entropy_value))
                
                val_bottlenecks, val_ground_truths = util.get_image_batch(flags.val_batch_size,
                                                                           'val',
                                                                     flags.image_dir,
                                                                     image_lists,
                                                                     decoded_image_tensor,
                                                                     jpeg_data_tensor,
                                                                     bottleneck_tensor,
                                                                     resized_input_tensor,
                                                                     sess)
                val_summary, val_accuracy = sess.run(
                    [merged, eval_step],
                    feed_dict = {bottleneck_input : val_bottlenecks,
                                 ground_truth_input : val_ground_truths})
                val_writer.add_summary(val_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (datetime.now(), i, val_accuracy * 100,
                                 len(val_bottlenecks)))

            if (flags.intermediate_store_frequency > 0 and (i % flags.intermediate_store_frequency == 0) and i > 0):
                intermediate_file_name = flags.intermediate_graph_dir + '/intermediate_' + str(i) + '.pb'
                tf.logging.info('Save intermediate result to : ' + intermediate_file_name)
                util.save_graph_to_file(sess, graph, intermediate_file_name, flags.final_tensor_name)

        test_bottlenecks, test_ground_truths = util.get_image_batch(flags.test_batch_size,
                                                                    'test',
                                                                    flags.image_dir,
                                                                    image_lists,
                                                                    decoded_image_tensor,
                                                                    jpeg_data_tensor,
                                                                    bottleneck_tensor,
                                                                    resized_input_tensor,
                                                                    sess)
        test_accuracy, preds = sess.run(
            [eval_step, pred],
            feed_dict={bottleneck_input : test_bottlenecks,
                       ground_truth_input : test_ground_truths})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %  (test_accuracy * 100, len(test_bottlenecks)))
        print(len(test_bottlenecks))

        util.save_graph_to_file(sess, graph, flags.fine_tuned_graph, flags.final_tensor_name)

    #with gfile.FastGFile(flags.labels, 'w') as f:
    #    f.write('\n'.join(image_lists.keys()) + '\n')
        
    with open(flags.operations, 'w') as f:
        for op in graph.get_operations():
            f.write(op.name+'\n')
        
