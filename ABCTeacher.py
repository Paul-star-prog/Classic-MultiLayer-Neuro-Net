# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:16:08 2020

@author: Павел Ермаков
"""
from abc import ABCMeta, abstractmethod
class ABCTeacher():
    __metaclass__=ABCMeta
    
    def __init__(self,nn):
        self.nn = nn
    @abstractmethod
    def _train_method():
        pass
    @abstractmethod
    def train():
        pass