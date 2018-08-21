#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/21
import numpy as np
import tensorflow as tf

from kge.model import BaseModel


class TransE(BaseModel):
    """This class implements transE model."""

    def build_graph(self):
        super(TransE, self).build_graph()
        with tf.name_scope('distance'):
            if self.params.score_func.lower() == 'l1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(self.h + self.r - self.t), axis=1)  # positive sample dissimilarity (energy)
                score_neg = tf.reduce_sum(tf.abs(self.h_neg + self.r - self.t_neg), axis=1)  # negative sample dissimilarity (energy)
            elif self.params.score_func.lower() == 'l2':  # L2 score
                score_pos = tf.sqrt(tf.reduce_sum(tf.square(self.h + self.r - self.t), axis=1))
                score_neg = tf.sqrt(tf.reduce_sum(tf.square(self.h_neg + self.r - self.t_neg), axis=1))
            self.predict = score_pos
            self.loss = tf.reduce_sum(tf.maximum(0.0, self.params.margin + score_pos - score_neg), name='max_margin_loss')
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

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

    def train(self, sess):
        # self._check_norm(sess=sess)
        return sess.run([self.loss, self.train_op, self.merge])

    def evaluation(self):
        pass

    def _check_norm(self, sess):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=sess)
        relation_embedding = self.relation_embedding.eval(session=sess)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))