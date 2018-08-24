#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/24
"""This module implements STransE model.
References:
    STransE: a novel embedding model of entities and relationships in knowledge bases, 2016
"""
import tensorflow as tf

from kge.model import BaseModel


class STransE(BaseModel):

    def _score_func(self, h, r, t):
        """f_r(h,t) = |Mr1*h+r-Mr2*t| constraints on the norm <=1"""
        # Projection matrix Mr, shape (k, k), initialize with identity matrix.
        self.Mr1 = tf.get_variable("Mr1", [self.k, self.k], initializer=tf.initializers.identity(gain=0.1))
        self.Mr1 = tf.tile(tf.expand_dims(self.Mr1, 0), [self.b, 1, 1])  # (b, k, k)
        self.Mr2 = tf.get_variable("Mr2", [self.k, self.k], initializer=tf.initializers.identity(gain=0.1))
        self.Mr2 = tf.tile(tf.expand_dims(self.Mr2, 0), [self.b, 1, 1])  # (b, k, k)

        h = tf.expand_dims(h, axis=2)  # (b, k) -> (b, k, 1)
        t = tf.expand_dims(t, axis=2)  # (b, k) -> (b, k, 1)
        dis = tf.squeeze(tf.matmul(self.Mr1, h), axis=2) + r + tf.squeeze(tf.matmul(self.Mr2, t), axis=2)
        if self.params.score_func.lower() == 'l1':  # L1 score
            score = tf.reduce_sum(tf.abs(dis), axis=1)
        elif self.params.score_func.lower() == 'l2':  # L2 score
            score = tf.sqrt(tf.reduce_sum(tf.square(dis), axis=1))

        return score