#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/21
"""This module implements transE model.
References:
    Translating Embeddings for Modeling Multi-relational Data, 2013
    """
import numpy as np
import tensorflow as tf

from kge.model import BaseModel


class TransE(BaseModel):

    def _score_func(self, h, r, t):
        with tf.name_scope('score'):
            if self.params.score_func.lower() == 'l1':  # L1 score
                score = tf.reduce_sum(tf.abs(h + r - t), axis=1)
            elif self.params.score_func.lower() == 'l2':  # L2 score
                score = tf.sqrt(tf.reduce_sum(tf.square(h + r - t), axis=1))
        return score

    def evaluate(self):
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + self.r - self.t  # broadcasting
            distance_tail_prediction = self.h + self.r - self.entity_embedding
        with tf.name_scope('rank'):
            if self.params.score_func.lower() == 'l1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(distance_head_prediction), axis=1), k=self.params.entity_size)
                _, idx_tail_prediction = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1), k=self.params.relation_size)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(
                    tf.reduce_sum(tf.square(distance_head_prediction), axis=1), k=self.params.entity_size)
                _, idx_tail_prediction = tf.nn.top_k(
                    tf.reduce_sum(tf.square(distance_tail_prediction), axis=1), k=self.params.relation_size)
        return idx_head_prediction, idx_tail_prediction

    def evaluation(self):
        pass

    def _check_norm(self, sess):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=sess)
        relation_embedding = self.relation_embedding.eval(session=sess)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))