# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:58:30 2020

@author: Павел Ермаков
"""
from ABCNeuralNetwork import ABCNeuralNetwork

class MultilayerNeuralNetwork(ABCNeuralNetwork):
    """
    Класс,реализующий многослойную нейронную сеть прямого распротсранения
    """
    def add_layer(self, layer):
        """
        Функция, реализующая добавление слоя в нейронную сеть
        :layer: слой
        """
        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Функция, реализующая нейронную сеть прямого распространения
        Вход:
        :X: входные значения сети
        Выход:
        :выход сети
        """
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        """
        Функция, вызывающая результат сети прямого распространения
        Вход:
        :X: входне значения сети
        Выход:
        :выход сети
        """
        return self.feed_forward(X)
