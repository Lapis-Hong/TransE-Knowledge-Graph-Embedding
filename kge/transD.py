#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/24
"""This module implements TransD model.
References:
    Knowledge Graph Embedding via Dynamic Mapping Matrix, 2015
"""
import tensorflow as tf

from kge.model import BaseModel


class TransD(BaseModel):

    def __init__(self, iterator, params):
        super(TransD, self).__init__(iterator, params)
        self.d = params.relation_embedding_dim  # overwrite self.d

    def _score_func(self, h, r, t):
        """f_r(h,t) = (I+r_p*h_p)*h +r -(I+r_p*t_p)*t """
        h = tf.expand_dims(h, 2)  # (b, k, 1)
        t = tf.expand_dims(t, 2)  # (b, k, 1)
        r = tf.expand_dims(r, 2)  # (b, k, 1)
        # Projection vectors
        rp = tf.get_variable("rp", [1, 1, self.d])
        rp = tf.tile(rp, [self.b, 1, 1])
        hp = tf.get_variable("hp", [1, 1, self.k])
        hp = tf.tile(hp, [self.b, 1, 1])
        tp = tf.get_variable("tp", [1, 1, self.k])
        tp = tf.tile(tp, [self.b, 1, 1])
        I = tf.eye(self.d, self.k)
        I = tf.tile(tf.expand_dims(I, 0), [self.b, 1, 1])  # (b, d, k)

        dis = tf.matmul(I+tf.matmul(rp, hp, transpose_a=True), h) + r - tf.matmul(I+tf.matmul(rp, tp, transpose_a=True), t)
        if self.params.score_func.lower() == 'l1':  # L1 score
            score = tf.reduce_sum(tf.abs(dis), axis=1)
        elif self.params.score_func.lower() == 'l2':  # L2 score
            score = tf.sqrt(tf.reduce_sum(tf.square(dis), axis=1))

        return score
