#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module contains some model utility functions."""
import codecs

import tensorflow as tf


def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def load_vocab(vocab_file):
    """load vocab from vocab file.
    Args:
        vocab_file: vocab file path
    Returns:
        vocab_table, vocab, vocab_size
    """
    vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file, default_value=0)
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab_table, vocab, vocab_size


def load_model(sess, ckpt):
    with sess.as_default():
        with sess.graph.as_default():
            init_ops = [tf.global_variables_initializer(),
                        tf.local_variables_initializer(), tf.tables_initializer()]
            sess.run(init_ops)
            # load saved model
            ckpt_path = tf.train.latest_checkpoint(ckpt)
            if ckpt_path:
                print("Loading saved model: " + ckpt_path)
            else:
                raise ValueError("No checkpoint found in {}".format(ckpt))
            # reader = tf.train.NewCheckpointReader(ckpt+'model.ckpt_0.876-580500')
            # variables = reader.get_variable_to_shape_map()
            # for v in variables:
            #     print(v)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)


