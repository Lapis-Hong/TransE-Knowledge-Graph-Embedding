#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/21
"""This module implements transH model.
References:
    Knowledge Graph Embedding by Translating on Hyperplanes, 2014
"""
import numpy as np
import tensorflow as tf

from kge.model import BaseModel


class TransH(BaseModel):
    """Model reflexive/one-to-many/many-to-one/many-to-many relations"""

    def _score_func(self, h, r, t):
        """f_r(h,t) = |(h-whw)+d_r-(t-wtw)|, w_r,d_r is orthogonal."""
        pass