# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:21:10 2020

@author: Павел Ермаков
"""
import numpy as np
from ABCTeacher import ABCTeacher

class BackPropagation(ABCTeacher):
          
    def _train_method(self, X, y, learning_rate):
        """
        Реализация алгоритма обратного распространения ошибки
        :X: вход сети
        :y: требуемые значения выходов сети
        :learning_rate: скорость обучения сети (0 - 1).
        """

        # Получение выходного значения сети
        output = self.nn.feed_forward(X)
        # Цикл по слоям сети в обратном порядке
        for i in reversed(range(len(self.nn._layers))):
            layer = self.nn._layers[i]

            # Если выходной слой
            if layer == self.nn._layers[-1]:
                # ошибка слоя
                layer.error = y - output
                # локальный градиент слоя
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self.nn._layers[i + 1]
                # ошибка слоя
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                # локальный градиент слоя
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Обновление весов 
        for i in range(len(self.nn._layers)):
            layer = self.nn._layers[i]
            # Здесь вход слоя:
            # либо выход предыдущего слоя
            # или сам вектор X входных значений сети (для 1-го скрытого слоя)
            input_to_use = np.atleast_2d(X if i == 0 else self.nn._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, X, Y, learning_rate, max_epochs):
        """
        Функция обучения сети с помощью метода обратного распространения ошибки
        Входы:
        :X: Входные параметры сети
        :Y: Требуемые значения функции
        :learning_rate: Скорость обучения (0 - 1)
        :max_epochs: Максимальное количество эпох обучения
        Выход:
        :mses: лист значений среднеквадратической ошибки
        """
        mses = []

        for i in range(max_epochs):
            for j in range(len(X)):
                # вызов функции обратного распространения ошибки
                self._train_method(X[j], Y[j], learning_rate)
            if i % 10 == 0:
                # выводить каждую 10 эпоху обучения средний квадрат ошибки 
                mse = np.mean(np.square(Y - self.nn.feed_forward(X)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
        return mses