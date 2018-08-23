#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/23
"""This module implements transR model.
References:
    Learning entity and relation embeddings for knowledge graph completion, 2015
"""
import tensorflow as tf

from kge.model import BaseModel


class TransR(BaseModel):
    """Model entities and relations in distinct spaces, 
    i.e., entity space and relation spaces, 
    and performs translation in relation space
    """
    def __init__(self, iterator, params):
        super(TransR, self).__init__(iterator, params)
        self.d = params.relation_embedding_dim  # overwrite self.d

    def _score_func(self, h, r, t):
        """f_r(h,t) = |M_r*h+r-M_r*t| constraints on the norm <=1"""
        # Projection matrix Mr, shape (d, k), initialize with identity matrix.
        self.Mr = tf.get_variable("Mr", [self.d, self.k], initializer=tf.initializers.identity(gain=0.1))
        self.Mr = tf.tile(tf.expand_dims(self.Mr, 0), [self.b, 1, 1])  # (b, k, d)
        h = tf.expand_dims(h, axis=2)  # (b, k) -> (b, k, 1)
        t = tf.expand_dims(t, axis=2)  # (b, k) -> (b, k, 1)
        dis = tf.squeeze(tf.matmul(self.Mr, h), axis=2) + r + tf.squeeze(tf.matmul(self.Mr, t), axis=2)
        if self.params.score_func.lower() == 'l1':  # L1 score
            score = tf.reduce_sum(tf.abs(dis), axis=1)
        elif self.params.score_func.lower() == 'l2':  # L2 score
            score = tf.sqrt(tf.reduce_sum(tf.square(dis), axis=1))

        return score


