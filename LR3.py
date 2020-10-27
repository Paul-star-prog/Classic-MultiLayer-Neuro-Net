# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:00:57 2020

@author: Павел Ермаков
"""
import matplotlib.pyplot as plt
from Layer import Layer
from NeuralNetwork import MultilayerNeuralNetwork
from BackPropogation import BackPropagation
from DataGenerator import DataGenerator


# экземпляр класса генерирования обучающей выборки
DG = DataGenerator(10,0,1)
# обучающая выборка
TrainData = DG.train_data()
# требуемые значения выхода
Zout = DG.fcn_value_data(TrainData)
# тестовая выборка
TestData1 = DG.test_data(1)
actual_y1 = DG.fcn_value_data(TestData1)
# Экземпляр класса нейронной сети
nn = MultilayerNeuralNetwork()
# Входной слой - 1-ый скрытый 
nn.add_layer(Layer(3, 10, 'sigmoid'))
# 1-ый - 2-ой скрытый слой
nn.add_layer(Layer(10, 5, 'sigmoid'))
# 2-ой - 3-ий скрытый слой
nn.add_layer(Layer(5, 3, 'sigmoid'))
# 3-ий скрытый слой - выходной слой
nn.add_layer(Layer(3, 1, 'sigmoid'))
# Экземпляр класса учителя
teacher = BackPropagation(nn)
# Обучение сети
errors = teacher.train(TrainData, Zout, 0.5, 4000)
# график Y_out
fig = plt.figure(figsize=(10,3))
fig1 = fig.add_subplot(121)
plt.plot(nn.predict(TestData1))
plt.plot(actual_y1)
plt.legend(['Approx_Y','True_Y'])
plt.xlabel('Point Count')
plt.ylabel('Y_out')
plt.title('Test Approx')
plt.ylim(0, 1)
# график MSE
fig2 = fig.add_subplot(122)
plt.plot(errors)
plt.xlabel('Epoch x10')
plt.ylabel('MSE')
plt.title('MSE Value')