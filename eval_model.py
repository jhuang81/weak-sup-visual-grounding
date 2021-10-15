import argparse
import os
import sys

import numpy as np
import tensorflow as tf
# Disable deprecation warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from dataset_utils import DatasetLoader
from nce_distill_model import get_input_variables, get_batch, lstm_cpc_model 

FLAGS = None


def eval_main(_):
    # Load data.
    data_loader = DatasetLoader(FLAGS.region_feat_path, FLAGS.phrase_feat_path, FLAGS.glove_path, split='eval')

    input_variables = get_input_variables(data_loader)
    for var_name in input_variables:
        input_variables[var_name] = tf.placeholder(*(input_variables[var_name]))
    input_variables['train_phase'] = tf.placeholder(tf.bool)
    FLAGS.num_region_proposals = data_loader.num_rp
    FLAGS.batch_size = 1
    input_variables['args'] = FLAGS
    # Override to suport variable #phrase per image.
    NUM_T = 10
    DIM_T = 300
    DIM_R = data_loader.dim_r
    NUM_R = data_loader.num_rp
    input_variables['region_feats'] = tf.placeholder(tf.float32, [None, DIM_R])
    input_variables['token_feats'] = tf.placeholder(tf.float32, [None, NUM_T, DIM_T])
    input_variables['lstm_mask'] = tf.placeholder(tf.bool)
    # Not used in eval.
    input_variables['distill_labels'] = tf.placeholder(tf.float32)
    input_variables['alignment_mask'] = tf.placeholder(tf.bool) 
    input_variables['alignment_gt'] = tf.placeholder(tf.float32)

    # Setup testing operation.
    NUM_R = data_loader.num_rp
    _, (_, _, (value_embed, phrase_embed)) = lstm_cpc_model(**input_variables)
    value_embed = tf.reshape(value_embed, [NUM_R, 512])
    phrase_embed = tf.reshape(phrase_embed, [-1, 512])
    # Transpose to make it consistent with the similarity models.
    similarity = tf.matmul(value_embed,
                           tf.transpose(phrase_embed))  # 200 * 16

    with tf.Session() as sess:
        # Restore latest checkpoint or the given MetaGraph.
        if FLAGS.restore_path.endswith('.meta'):
            ckpt_path = FLAGS.restore_path.replace('.meta', '')
        else:
            ckpt_path = tf.train.latest_checkpoint(FLAGS.restore_path)

        print('Restoring checkpoint', ckpt_path)
        tensors = tf.global_variables()
        saver = tf.train.Saver(tensors)
        saver.restore(sess, ckpt_path)
        print('Done')

        # For testing and validation,vthere will be multiple image-sentence batch,
        # but the result should be independent of batch size.
        NUM_PAIRS = len(data_loader.example_inds)
        print('Evaluating %d pairs' % NUM_PAIRS)
        assert(FLAGS.batch_size == 1)
        ks = [1, 5, 10]
        correct_count, total_count = np.zeros(len(ks), dtype=np.int32), 0

        for i in range(NUM_PAIRS):
            input_values = get_batch(data_loader, i)
            input_values['train_phase'] = False
            feed_dict = {input_variables[name] : input_values[name]
                         for name in input_variables if name in input_values}
            [similarity_val] = sess.run([similarity], feed_dict = feed_dict)  # 200 * 16
            sorted_region_index = np.argsort(similarity_val.T, axis=-1)  # 16 * 200
            query_index = sorted_region_index[:, -ks[-1]:]  # 16 * 10
            num_phrase = int(np.sum(input_values['gt_boxes'][0, :, -1] > 0))
            if num_phrase == 0:
                continue
            total_count += num_phrase
            is_correct = np.array([input_values['alignment_gt'][0, query_index[k, :], k]
                                   for k in range(num_phrase)])
            correct_count += np.array([np.sum(np.max(is_correct[:, -k:], axis=-1)) for k in ks])
        for i in range(len(ks)):
            print('Recall@%d %f (%d/%d)' % (ks[i], correct_count[i] * 1.0 / total_count, correct_count[i], total_count))


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--region_feat_path', type=str,
                        help='Path to the region feature hdf5 file.')
    parser.add_argument('--phrase_feat_path', type=str,
                        help='Path to the phrase feature hdf5 file.') 
    parser.add_argument('--glove_path', type=str,
                        help='Path to the glove embedding hdf5 file.')
    parser.add_argument('--restore_path', type=str,
                        help='Directory for restoring the newest checkpoint or \
                              path to a restoring checkpoint MetaGraph file.')
    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=eval_main, argv=[sys.argv[0]] + unparsed)
