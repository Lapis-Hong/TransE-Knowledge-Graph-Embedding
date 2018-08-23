#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
"""This module for model training."""
import os
import datetime

import tensorflow as tf

from config import FLAGS
from kge import *
from dataset import get_iterator
from utils import print_args, load_vocab


def train():
    # Training
    with tf.Session() as sess:
        init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()]
        sess.run(init_ops)
        writer = tf.summary.FileWriter("summary", sess.graph)  # graph

        for epoch in range(FLAGS.max_epoch):
            sess.run(iterator.initializer)
            model.train(sess)
            if not os.path.exists(FLAGS.model_dir):
                os.mkdir(FLAGS.model_dir)
            save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
            model.save(sess, save_path)

            print('-----Start training-----')
            epoch_loss = 0.0
            step = 0
            while True:
                try:
                    batch_loss, _, summary = model.train(sess)
                    epoch_loss += batch_loss
                    step += 1
                    writer.add_summary(summary)
                except tf.errors.OutOfRangeError:
                    print('-----Finish training an epoch avg epoch loss={}-----'.format(epoch_loss / step))
                    break
                # show train batch metrics
                if step % FLAGS.stats_per_steps == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}\tepoch {:2d}\tstep {:3d}\ttrain loss={:.6f}'.format(
                        time_str, epoch + 1, step, batch_loss))

            if (epoch+1) % FLAGS.save_per_epochs == 0:
                if not os.path.exists(FLAGS.model_dir):
                    os.mkdir(FLAGS.model_dir)
                save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                model.save(sess, save_path)
                print("Epoch {}, saved checkpoint to {}".format(epoch+1, save_path))


if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Params Preparation
    print_args(FLAGS)
    entity_table, entity, entity_size = load_vocab(FLAGS.entity_vocab)
    relation_table, _, relation_size = load_vocab(FLAGS.relation_vocab)
    FLAGS.entity_size = entity_size
    FLAGS.relation_size = relation_size

    # Model Preparation
    mode = tf.estimator.ModeKeys.TRAIN
    iterator = get_iterator(
        FLAGS.data_file, entity, entity_table, relation_table, FLAGS.batch_size, shuffle_buffer_size=FLAGS.shuffle_buffer_size)
    if FLAGS.model_name.lower() == "transe":
        model = TransE(iterator, FLAGS)
    elif FLAGS.model_name.lower() == "distmult":
        model = DISTMULT(iterator, FLAGS)
    elif FLAGS.model_name.lower() == "transh":
        model = TransH(iterator, FLAGS)
    elif FLAGS.model_name.lower() == "transr":
        model = TransR(iterator, FLAGS)

    model.build_graph()  # build graph
    train()
