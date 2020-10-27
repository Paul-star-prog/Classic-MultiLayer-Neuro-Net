# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:06:13 2020

@author: Павел Ермаков
"""
from abc import ABCMeta, abstractmethod
class ABCDataGenerator():
    __metaclass__=ABCMeta
    
    def __init__(self,n,x_low,x_up):
        self.n = n
        self.x_low = x_low
        self.x_up = x_up
    @abstractmethod
    def train_data():
        pass
    @abstractmethod
    def test_data():
        pass
    @abstractmethod
    def fcn_value_data():
        pass
    @abstractmethod
    def _my_fcn():
        pass