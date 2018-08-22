#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/21
"""This module for generating negative triple samples. (Deprecated)"""
import random


def generate_negative_samples(infile, outfile):
    """Infile format: each line is a triple of (h, t, r)."""
    head_entities, tail_entities = set(), set()
    triples = []
    for line in open(infile):
        h, t, r = line.strip().split('\t')
        head_entities.add(h)
        tail_entities.add(t)
        triples.append([h, t, r])

    head_entities = list(head_entities)
    tail_entities = list(tail_entities)
    with open(outfile, "w") as fo:
        for h, t, r in triples:
            head_neg = h
            tail_neg = t
            prob = random.random()
            if prob > 0.5:
                head_neg = random.choice(head_entities)
            else:
                tail_neg = random.choice(tail_entities)
            fo.write('\t'.join([head_neg, tail_neg, r])+'\n')


if __name__ == '__main__':
    generate_negative_samples("FB15k/train.txt", "FB15k/train_neg.txt")
    generate_negative_samples("WN18/train.txt", "WN18/train_neg.txt")
