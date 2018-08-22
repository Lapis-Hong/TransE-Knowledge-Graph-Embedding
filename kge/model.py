#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/13
"""This module implements abstract model class for Knowledge-Graph-Embedding models."""
import abc
import math
from functools import reduce

import tensorflow as tf
from kge.model_utils import get_optimizer_instance


class BaseModel(object):
    """Abstract base class."""

    def __init__(self, iterator, params):
        """Initialize model, build graph.
        Args:
          iterator: instance of class BatchedInput, defined in dataset.  
          params: parameters.
        """
        self.iterator = iterator
        self.params = params
        self.k = params.embedding_dim  # embedding dimension
        self.b = params.batch_size  # batch_size
        self.scope = self.__class__.__name__  # instance class name
        self.build_graph()
        self._model_stats()  # print model statistics info

    def build_graph(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # embedding
            bound = 6 / math.sqrt(self.k)
            with tf.variable_scope('embedding'):
                self.entity_embedding = tf.get_variable(
                    name='entity',
                    shape=[self.params.entity_size, self.k],
                    initializer=tf.random_uniform_initializer(-bound, bound))
                tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
                self.relation_embedding = tf.get_variable(
                    name='relation',
                    shape=[self.params.relation_size, self.k],
                    initializer=tf.random_uniform_initializer(-bound, bound))
                tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

            with tf.name_scope('normalization'):
                self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, axis=1)
                self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, axis=1)

            with tf.name_scope('lookup'):
                self.h = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.h)
                self.t = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.t)
                self.r = tf.nn.embedding_lookup(self.relation_embedding, self.iterator.r)
                self.h_neg = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.h_neg)
                self.t_neg = tf.nn.embedding_lookup(self.entity_embedding, self.iterator.t_neg)

            score_pos = self._score_func(self.h, self.r, self.t)
            score_neg = self._score_func(self.h_neg, self.r, self.t_neg)
            self.predict = score_pos
            self.loss = tf.reduce_sum(tf.maximum(0.0, self.params.margin + score_pos - score_neg),
                                      name='max_margin_loss')
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = get_optimizer_instance(self.params.optimizer, self.params.learning_rate)
            self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    @abc.abstractmethod
    def _score_func(self, h, r, t):
        """Score / energy functions f_r(h, t), must be implemented in subclasses.
        Args:
            h, r, t: Tensor (b, k)
        Returns:
            Score tensor (b, 1)
        """
        pass

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))

    def train(self, sess):
        return sess.run([self.loss, self.train_op, self.merge])

    # def eval(self, sess):
    #     return sess.run([self.loss, self.pred])
    #
    # def predict(self, sess):
    #     return sess.run([self.pred])

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, path, global_step=self.global_step.eval())





