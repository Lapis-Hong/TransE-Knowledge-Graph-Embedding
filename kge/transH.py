#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/21
"""This module implements transH model.
References:
    Knowledge Graph Embedding by Translating on Hyperplanes, 2014
"""
#TODO: add norm constraints
import tensorflow as tf

from kge.model import BaseModel


class TransH(BaseModel):
    """Model reflexive/one-to-many/many-to-one/many-to-many relations"""

    def _score_func(self, h, r, t):
        """f_r(h,t) = |(h-whw)+d_r-(t-wtw)|, w_r,d_r is orthogonal."""
        self.w = tf.get_variable("Mr", [1, 1, self.k])
        self.w = tf.tile(self.w, [self.b, 1, 1])  # (b, 1, k)
        h = tf.expand_dims(h, axis=2)  # (b, k) -> (b, k, 1)
        t = tf.expand_dims(t, axis=2)  # (b, k) -> (b, k, 1)
        h_v = tf.squeeze(h, axis=2) - tf.squeeze(tf.matmul(self.w, tf.matmul(h, self.w)), axis=1)
        t_v = tf.squeeze(t, axis=2) - tf.squeeze(tf.matmul(self.w, tf.matmul(t, self.w)), axis=1)
        dis = h_v + r - t_v
        if self.params.score_func.lower() == 'l1':  # L1 score
            score = tf.reduce_sum(tf.abs(dis), axis=1)
        elif self.params.score_func.lower() == 'l2':  # L2 score
            score = tf.sqrt(tf.reduce_sum(tf.square(dis), axis=1))

        return score
