#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/14
"""This module contains several models for Knowledge Graph Embedding

All model classes must inherit class `BaseModel` (defined in model.py) 
"""

# import selected Classes into the package level so they can be convieniently imported from the package.
# use from model import TransE instead of from model.transE import TransE
from kge.transE import TransE
from kge.distmult import DISTMULT
# from model import *
__all__ = ["TransE", "DISTMULT"]

