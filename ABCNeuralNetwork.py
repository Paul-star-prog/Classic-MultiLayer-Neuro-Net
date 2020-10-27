# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:50:44 2020

@author: Павел Ермаков
"""
from abc import ABCMeta
class ABCNeuralNetwork():
    __metaclass__=ABCMeta
    def __init__(self):
        self._layers = []
