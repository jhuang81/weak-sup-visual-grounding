# Compiled with TensorFlow 1.15.

import argparse
import json
import sys

import numpy as np
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
import tensorflow as tf

from bbox import bbox_overlaps
from dataset_utils import DatasetLoader
from lstm_encoder import Encoder

# For distillation only.
# COCO
# from dataset_utils import _COCO_CLASSES as CLASS_NAME
# OI
from oiv2_classes import CLASS_NAME


# Constants for distillation.
BG_AND_CLASSES = ['__background__'] + CLASS_NAME
# Enable distillation with phrase to class label mapping computed offline.
PHRASE_TO_CLASS_INDEX = {}
GLOBAL_VARS = {}


# Constants for command line flags.
FLAGS = None


def feedforward_net(features, layer_out_dims=[1024, 512], scope_in=''):
    """
        Encodes features into lower dimensional embeddings.
    """
    with tf.variable_scope(scope_in) as scope:
        outputs = features
        for i in range(len(layer_out_dims) - 1):
            outputs = tf.compat.v1.layers.dense(
                    inputs=outputs, units=layer_out_dims[i], activation=tf.nn.relu, name='fc%d' % i)
        outputs = tf.compat.v1.layers.dense(
                    inputs=outputs, units=layer_out_dims[-1], activation=None,
                    name='fc%d' % (len(layer_out_dims) - 1))
    return outputs


def cpc_loss(logits, labels, weights=None):
    cpc_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(labels), logits=logits)
    if weights is not None:
        cpc_loss = weights * cpc_loss
    cpc_loss = tf.reduce_mean(cpc_loss)
    return cpc_loss
 

def lstm_cpc_model(region_feats, token_feats, alignment_mask, lstm_mask, train_phase,
                   alignment_gt, distill_labels, region_ious, args):
    fc_dim = 1024
    embed_dim = 512 
    lstm_dim = 300
    B = args.batch_size
    NUM_R = args.num_region_proposals
    NUM_P = 16
 
    # Enocde phrase features.
    with tf.variable_scope('phrase'):
        lstm_encoder = Encoder(lstm_dim, 10)
        phrase_feats = lstm_encoder.encode(token_feats, lstm_mask)
    # Compute emebeddings.
    region_embed = feedforward_net(region_feats, layer_out_dims=[fc_dim, embed_dim],
                                   scope_in='region')
    phrase_embed = feedforward_net(phrase_feats, layer_out_dims=[fc_dim, embed_dim],
                                   scope_in='phrase')
    # L2 normalize for margin.
    region_embed = tf.nn.l2_normalize(region_embed, axis=-1)
    phrase_embed = tf.nn.l2_normalize(phrase_embed, axis=-1)
    if args.restore_path:
      # Return if evaluation.
      return (None, None), (None, None, (region_embed, phrase_embed))

    # Loss.
    similarity = tf.matmul(region_embed, tf.transpose(phrase_embed))  # 200b * 16b
    phrase_region_similarity = tf.transpose(similarity)       # 16b * 200b
    # Log index of the selected region for computing training gournding accuracy.
    _, region_indices = tf.nn.top_k(tf.reshape(
            phrase_region_similarity, [B*NUM_P, B, NUM_R]), k=1) # 16b * b * 1
    region_indices = tf.reshape(region_indices, [B*NUM_P, B])

    # Remove paddings.
    alignment_mask = tf.reshape(alignment_mask, [B * NUM_P])  # 16b
    phrase_region_similarity = tf.boolean_mask(
            phrase_region_similarity, alignment_mask)         # P * 200b
    phrase_region_similarity = tf.reshape(
            phrase_region_similarity, [-1, B, NUM_R])
    # Estimate MI(I, p) as max{MI(r_i, p)}.
    phrase_region_similarity = tf.reduce_max(
            phrase_region_similarity, axis=-1)  # P * b
    indices = tf.where(alignment_mask)  # P * 1
    phrase_region_similarity = tf.scatter_nd(
            indices=indices, updates=phrase_region_similarity,
            shape=[NUM_P*B, B])
    phrase_region_similarity = tf.reshape(
            phrase_region_similarity, [B, NUM_P, B])
    phrase_region_similarity = phrase_region_similarity * 2.0
    logits = tf.reduce_sum(phrase_region_similarity, axis=1)  # b * b
    labels = tf.eye(B)  # b * b
    loss = cpc_loss(logits, labels)

    if len(PHRASE_TO_CLASS_INDEX) == 0:
        return (loss, tf.zeros(1)), (logits, region_indices, (region_embed, phrase_embed))

    # Distillation.
    distill_mask = tf.reduce_any(
        ~tf.math.equal(distill_labels, tf.zeros_like(distill_labels)), axis=-1)  # 16b
    K = 8
    _, nn_index = tf.nn.top_k(region_ious, k=K)  # b * 200 * K
    pos_pair_mask = tf.reshape(tf.tile(tf.eye(B, dtype=tf.bool), [1, NUM_P]), [B * NUM_P, B])  # 16b * b
    pos_region_indices = tf.boolean_mask(region_indices, pos_pair_mask)   # 16b
    pos_region_indices = tf.reshape(pos_region_indices, [B, NUM_P])       # b * 16
    nn_index = tf.concat(
        [tf.gather(nn_index[i], pos_region_indices[i]) for i in range(B)], axis=0)  # 16b * K
    row_index = tf.tile(tf.reshape(tf.range(B * NUM_P), [-1, 1]), [1, K])  # 16b * K
    nn_index = tf.stack([row_index, nn_index], axis=-1)  # 16b * K *2
    distill_logits = tf.reshape(tf.transpose(similarity), [B*NUM_P, B, NUM_R])  # 16b * b * 200
    distill_logits = tf.boolean_mask(distill_logits, pos_pair_mask)  # 16b * 200
    distill_logits = tf.gather_nd(distill_logits, nn_index)  # 16b * K
    distill_labels = tf.gather_nd(distill_labels, nn_index)  # 16b * K
    distill_logits = tf.boolean_mask(distill_logits, distill_mask)  # P * K
    distill_labels = tf.boolean_mask(distill_labels, distill_mask)  # P * K

    # Normalizes detector class predictions by shifting or scaling.
    # Due to the different number of classes, the logits need to be normalized differently.
    if len(CLASS_NAME) == 80:  # COCO
      distill_labels = distill_labels / 2.0
    elif len(CLASS_NAME) == 545:  # OI
      distill_labels = distill_labels - tf.reduce_mean(distill_labels, axis=-1, keepdims=True)
    else:
      raise NotImplementedError
    distill_labels = tf.nn.softmax(distill_labels, axis=-1)
    distill_loss = cpc_loss(logits=distill_logits, labels=distill_labels) 
    # Ramping factor to be tuned based on the distillation features used.
    ramping_factor = tf.minimum(0.001 + tf.cast(GLOBAL_VARS['global_step'] // 500, tf.float32), 3.0)
    # ramping_factor = tf.minimum(0.001 + tf.cast(GLOBAL_VARS['global_step'] // 200, tf.float32), 3.0)
    loss = 1.0 * loss + ramping_factor * distill_loss
    return (loss, distill_loss), (logits, region_indices, (region_embed, phrase_embed))


def setup_train_model(region_feats, token_feats, alignment_mask, lstm_mask, train_phase,
                      alignment_gt, distill_labels, region_ious, args):
    (loss, distillation_loss), tensors = lstm_cpc_model(
            region_feats, token_feats, alignment_mask, lstm_mask, train_phase,
            alignment_gt, distill_labels, region_ious, args)
    metrics = [tf.constant(0.0) for _ in range(3)]
    # Evaluates training/eval objective.
    logits, region_indices, _ = tensors
    pos_mask = tf.eye(tf.shape(logits)[0], dtype=tf.bool)
    pos_logits = tf.reshape(tf.boolean_mask(logits, pos_mask),
                            [tf.shape(logits)[0], 1])  # k
    neg_logits = tf.reshape(tf.boolean_mask(logits, ~pos_mask),
                            [tf.shape(logits)[0], -1]) # k * (k-1)
    # Precision@1
    metrics[0] = tf.reduce_mean(
            tf.cast(
                pos_logits > tf.reduce_max(neg_logits, axis=-1, keepdims=True),
                tf.float32))
    # Avg #(pos > neg).
    metrics[1] = tf.reduce_mean(tf.cast(pos_logits > neg_logits, tf.float32))
    # Grounding accuracy.
    B = args.batch_size
    NUM_P = 16
    NUM_R = args.num_region_proposals
    pos_pair_mask = tf.reshape(tf.tile(tf.eye(B, dtype=tf.bool), [1, NUM_P]), [B * NUM_P, B])  # 16b * b
    pos_region_indices = tf.boolean_mask(region_indices, pos_pair_mask)   # 16b
    phrase_indices = tf.range(B * NUM_P)  # 16b
    pos_region_indices = tf.stack([phrase_indices, pos_region_indices], axis=-1)  # 16b * 2
    alignment_gt = tf.reshape(tf.transpose(alignment_gt, [0, 2, 1]), [B*NUM_P, NUM_R])  # 16b * 200
    alignment_mask = tf.reshape(alignment_mask, [B * NUM_P])  # 16b
    metrics[2] = tf.reduce_mean(
            tf.boolean_mask(tf.gather_nd(alignment_gt, pos_region_indices), alignment_mask))
    return (loss, distillation_loss) + tuple(metrics)


# Functions for model training.


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_input_variables(data_loader):
  B = FLAGS.batch_size if FLAGS is not None else 1
  DIM_R = data_loader.dim_r
  NUM_R = data_loader.num_rp
  NUM_P = data_loader.max_phrase_per_sentence
  input_variables = data_loader.get_input_variables(batch_size=B)
  input_variables['region_ious'] = (tf.float32, (B, NUM_R, NUM_R))
  input_variables['distill_labels'] = (tf.float32, (B*NUM_P , NUM_R))
  return input_variables


def get_batch(data_loader, batch_index):
  DIM_R = data_loader.dim_r
  NUM_R = data_loader.num_rp
  B = FLAGS.batch_size if FLAGS is not None else 1
  NUM_P = data_loader.max_phrase_per_sentence
  C = len(CLASS_NAME)
  NUM_T = data_loader.max_token_per_phrase
  input_values = data_loader.get_batch(batch_index, B) 
  region_feats = input_values['region_feats']  # (B * NUM_R) * DIM_R
  query_boxes = input_values['query_boxes'].reshape([B, NUM_R, 4])

  enable_distill = len(PHRASE_TO_CLASS_INDEX) > 0 and data_loader.split == 'train'
  distill_labels = np.zeros([B, NUM_P, NUM_R])
  if enable_distill:
    region_ious = np.stack([
        bbox_overlaps(query_boxes[i].reshape([-1, 4]),
        query_boxes[i].reshape([-1, 4])) for i in range(B)])   # B * NUM_R * NUM_R
    input_values['region_ious'] = region_ious
  else:
    input_values['region_ious'] = np.zeros([B, NUM_R, NUM_R])

  if enable_distill:
    # Compute region logits.
    # OID
    region_logits_oid = input_values['region_logits']
    # Do NOT use the logits due to numerical stability issue.
    region_logits = softmax(region_logits_oid, axis=-1)
    # For each phrase, find a class label and copy over logits if exists.
    for i in range(B):
      for p in range(NUM_P):
        phrase = ' '.join(input_values['phrases'][i, p]).strip()
        if not phrase or phrase not in PHRASE_TO_CLASS_INDEX:
          continue
        cls_index = PHRASE_TO_CLASS_INDEX[phrase]
        if cls_index >= region_logits.shape[-1]:  # For res101-oid combined features
          continue
        cls_logits = region_logits[i*NUM_R:(i+1)*NUM_R, cls_index]  # 200
        distill_labels[i, p] = cls_logits
  input_values['distill_labels'] = np.reshape(distill_labels, [B*NUM_P, NUM_R])  # 16B * 200
  return input_values


def main(args):
    global FLAGS
    FLAGS = args

    # Enable distillation.
    if FLAGS.phrase_to_label_json:
        phrase_to_label = json.load(open(FLAGS.phrase_to_label_json, 'r'))
        PHRASE_TO_CLASS_INDEX.update({
            p : BG_AND_CLASSES.index(phrase_to_label[p]) for p in phrase_to_label})
        print('Enable distillation: #mapped phrases=%d' % len(PHRASE_TO_CLASS_INDEX))
    else:
        print('NO distillation.')


    # Load data.
    data_loader = DatasetLoader(FLAGS.region_feat_path, FLAGS.phrase_feat_path, FLAGS.glove_path)
    steps_per_epoch = data_loader.example_inds.size // FLAGS.batch_size
    num_steps = steps_per_epoch * FLAGS.max_num_epoch + 1
    print('#steps: %d' % num_steps, '#steps per epoch: %d' % steps_per_epoch)
    print('batch size: %d' % FLAGS.batch_size, 'smaple size: %d' % FLAGS.sample_size)
    print('learning rate %.6f' % FLAGS.init_learning_rate)

    # Setup placeholders for input variables.
    input_variables = get_input_variables(data_loader)
    for var_name in input_variables:
        input_variables[var_name] = tf.placeholder(*(input_variables[var_name]))
    input_variables['train_phase'] = tf.placeholder(tf.bool)
    input_variables['args'] = FLAGS

    # Setup training operation.
    global_step = tf.Variable(0, trainable=False)
    # GLOBAL_VARS are used in model initialization, hence need to be decleared
    # before calling setup_train_model.
    GLOBAL_VARS['global_step'] = global_step
    losses = setup_train_model(**input_variables)

    # Get model variables.
    tensors = tf.global_variables()
    print([t.name for t in tensors])

    # Setup optimizer.
    decay_steps = FLAGS.decay_steps if FLAGS.decay_steps > 0 else steps_per_epoch
    learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step,
                                               decay_steps, FLAGS.decay_rate, staircase=True)
    optim = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # ONLY optimize the first loss term.
        train_step = optim.minimize(losses[0], global_step=global_step)

    # Setup pretrained model saver.
    pretrained_vars = {t.name.replace('pretrained/', '').rstrip(':0') : t for t in tensors
                       if 'pretrained' in t.name}
    pretrained_saver = tf.train.Saver(var_list=pretrained_vars) if pretrained_vars else None
    # Setup model saver.
    model_vars = [t for t in tensors if 'pretrained' not in t.name]
    print('#pretrained_vars=%d, #model_vars=%d' % (len(pretrained_vars), len(model_vars)))
    saver = tf.train.Saver(var_list=model_vars, max_to_keep=20)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.restore_path:
            if FLAGS.restore_path.endswith('.meta'):
                ckpt_path = FLAGS.restore_path.replace('.meta', '')
            else:
                ckpt_path = tf.train.latest_checkpoint(FLAGS.restore_path)
            print('Restoring checkpoint', ckpt_path)
            saver.restore(sess, ckpt_path)
            print('Done')
        if pretrained_saver and FLAGS.pretrained_model_path:
            if FLAGS.pretrained_model_path.endswith('.meta'):
                pretrained_ckpt_path = FLAGS.pretrained_model_path.replace('.meta', '')
            print('Restoring pretrained checkpoint', pretrained_ckpt_path)
            pretrained_saver.restore(sess, pretrained_ckpt_path)
            print('Done')
        print('#global_variables=', len(tf.global_variables()))
        avg_losses = np.zeros(len(losses))
        for i in range(num_steps):
            if i % steps_per_epoch == 0:
                # shuffle the indices.
                data_loader.shuffle_inds()
                # Reset to 0.
                avg_losses = np.zeros(len(losses))
            input_values = get_batch(data_loader, i % steps_per_epoch)
            input_values['train_phase'] = False   # False to turn off dropout.
            feed_dict = {input_variables[name] : input_values[name]
                         for name in input_variables if name in input_values}
            train_ops = (train_step,) + losses
            train_ops_val = sess.run(train_ops, feed_dict = feed_dict)
            losses_val = np.array(train_ops_val[1:])  # Exclude the first value which is returned by train_step.
            avg_losses = (losses_val + avg_losses * (i % steps_per_epoch)) / (i % steps_per_epoch + 1)
            if i % 50 == 0:
                print('Epoch: %d Step: %d Loss:' % (i // steps_per_epoch, i) +
                      ' %.3f' * len(avg_losses) % tuple(avg_losses))
            if (i % 500 == 0 or i % steps_per_epoch == 0) and i > 0:
                saver.save(sess, FLAGS.save_dir, global_step = global_step)
                print('Saved checkpoint at step %d' % i)
