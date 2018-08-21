#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module contains efficient data read and transform using tf.data API.

Data iterator for triplets (h, t, r) 
and corrupt sampling (with either the head or tail replaced by a random entity).


Input format:
    Train data: data file
        each line contains (h, t, r) triples separated by '\t'
"""
import collections
import random

import tensorflow as tf


class BatchedInput(
    collections.namedtuple(
        "BatchedInput", ("initializer", "h", "t", "r", "h_neg", "t_neg"))):
    pass


def _parse(line):
    """Parse train data."""
    cols_types = [[''], [''], ['']]
    return tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')


def get_iterator(data_file, entity, entity_table, relation_table, batch_size, shuffle_buffer_size=None):
    """Iterator for train and eval.
    Args:
        data_file: data file, each line contains (h, t, r) triple
        entity: list or tuple of all entities.
        entity_table: entity tf look-up table
        relation_table: relation tf look-up table
        shuffle_buffer_size: buffer size for shuffle
    Returns:
        BatchedInput instance
    """
    shuffle_buffer_size = shuffle_buffer_size or batch_size * 1000

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(_parse, num_parallel_calls=4)
    dataset = dataset.shuffle(shuffle_buffer_size)

    # corrupt sampling
    def sample():
        if random.random() < 0.5:
            return lambda h, t, r: (h, t, r, random.choice(entity), t)
        else:
            return lambda h, t, r: (h, t, r, h, random.choice(entity))

    dataset = dataset.map(sample())

    dataset = dataset.map(
        lambda h, t, r, h_neg, t_neg: (
            tf.cast(entity_table.lookup(h), tf.int32),
            tf.cast(entity_table.lookup(t), tf.int32),
            tf.cast(relation_table.lookup(r), tf.int32),
            tf.cast(entity_table.lookup(h_neg), tf.int32),
            tf.cast(entity_table.lookup(t_neg), tf.int32)
        ),
        num_parallel_calls=4)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ),
        padding_values=(0, 0, 0, 0, 0),
        drop_remainder=True).prefetch(2*batch_size)

    batched_iter = dataset.make_initializable_iterator()
    h, t, r, h_neg, t_neg = batched_iter.get_next()

    return BatchedInput(initializer=batched_iter.initializer, h=h, t=t, r=r, h_neg=h_neg, t_neg=t_neg)

