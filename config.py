#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module contains hyperparameters using tf.flags."""
import tensorflow as tf

# data params
tf.flags.DEFINE_string("data_file", 'data/FB15k/train.txt', "data dir")
tf.flags.DEFINE_string("entity_vocab", 'data/FB15k/entity.vocab', "entity vocab file")
tf.flags.DEFINE_string("relation_vocab", 'data/FB15k/relation.vocab', "relation vocab file")
tf.flags.DEFINE_integer("embedding_dim", 200, "embedding dim [200]")
tf.flags.DEFINE_integer("shuffle_buffer_size", 10000, "Shuffle buffer size")

# model params
tf.flags.DEFINE_string("model_type", "transE", "modell name, `transE` or `transH`.")
tf.flags.DEFINE_string("model_dir", "model", "model path")
tf.flags.DEFINE_float("margin", 1.0, "loss margin")
tf.flags.DEFINE_string("score_func", "l1", "score function type")

# tf.flags.DEFINE_float("l2_reg", 0.004, "l2 regularization weight [0.004]")

# training params
tf.flags.DEFINE_integer("batch_size", 32, "train batch size [64]")
tf.flags.DEFINE_integer("max_epoch", 100, "max epoch [100]")
tf.flags.DEFINE_float("learning_rate", 0.002, "init learning rate [adam: 0.002, sgd: 1.1]")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer, `adam` | `rmsprop` | `sgd` [adam]")
tf.flags.DEFINE_integer("stats_per_steps", 10, "show train info steps [100]")

# auto params, do not need to set
tf.flags.DEFINE_integer("entity_size", None, "entity vocabulary size")
tf.flags.DEFINE_integer("relation_size", None, "relation vocabulary size")


FLAGS = tf.flags.FLAGS
