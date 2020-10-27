# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:09:18 2020

@author: Павел Ермаков
"""

from ABCDataGenerator import ABCDataGenerator
class DataGenerator(ABCDataGenerator):
    """
    Класс,реализующий генерирование тестовой, тренировочной выборки
    """
    def __init__(self,n,x_low,x_up):
        ABCDataGenerator.__init__(self,n,x_low,x_up)
        self.arg = 3     
        
    def train_data(self):
        xs = []
        for i in range(self.arg):
            xs.append([])
        for number in range(self.n):
            for i in range(self.arg):
                xs[i].append(self.x_low + number/self.n)
        Xs = []
        for _x1 in xs[0]:
            for _x2 in xs[1]:
                for _x3 in xs[2]:
                    Xs.append([_x1,_x2,_x3])
        return Xs
    
    def test_data(self,delta):
        x_pred = []
        for i in range(self.arg):
            x_pred.append([]) 
        for number in range(self.n):
            for i in range(self.arg):
                x_pred[i].append(self.x_low + delta * number/self.n)
        Xs_pred = []
        for _x1, _x2,_x3 in zip(x_pred[0], x_pred[1], x_pred[2]):
            Xs_pred.append([_x1, _x2,_x3])
        return Xs_pred
    
    def fcn_value_data(self, Xs_pred):
        return [self._my_fcn(_x1, _x2,_x3) for _x1, _x2,_x3 in Xs_pred]
        
    def _my_fcn(self, x1, x2, x3):
        return [(x1**2 + x2**2 + x3**2 )]
