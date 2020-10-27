# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:31:04 2020

@author: Павел Ермаков
"""
import numpy as np
class Layer:
    """
    Класс,реализующий слой нейронной сети
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        Конструктор
        :n_input: вход слоя
        :n_neurons: количесвто нейронов в слое
        :activation: функция активации нейронов в слое
        :weights: весы в слое
        :bias: смещения слоя
        """
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        """
        Функция, вычисляющая значение функции активации конкретного слоя
        Вход:
        :x: вход слоя
        Выход:
        :return: значение функции активации
        """
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Функция, реализующая функцию активации нейрона
        Вход:
        :r: входное значение
        Выход:
        : акстивированное значение
        """

        # если не выбрана функция активации
        if self.activation is None:
            return r

        # гиперболический тангенс
        if self.activation == 'tanh':
            return np.tanh(r)

        # сигмоида
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        return r

    def apply_activation_derivative(self, r):
        """
        Функция, реализующая вычисление производной функции активации
        Вход:
        :r: входное значение
        Выход:
        :значение производной
        """

        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        return r