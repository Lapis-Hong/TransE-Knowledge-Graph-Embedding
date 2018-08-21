#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/21
import numpy as np
import tensorflow as tf

from kge.model import BaseModel


class TransH(BaseModel):
    """This class implements transE model."""