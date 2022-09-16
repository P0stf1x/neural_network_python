from math import exp
from random import uniform as random


class Neuron:
    def __init__(self, value=None):
        self.value = value
        self.delta = None
        self.backNeighbors = []  # 0 - нейрон, 1 - вес, 2 - дельта вес

    def initNeighbors(self, backneighbors):  # Получение списка соседних слоёв
        for i in range(len(backneighbors)):
            temp = [backneighbors[i], random(-1, 1), 0]
            self.backNeighbors.append(temp)

    def updateWeight(self, learningRate, momentum):
        for synopsis in self.backNeighbors:
            gradient = self.delta * synopsis[0].value  # градиент
            deltaWeight = learningRate * gradient + momentum * synopsis[2]  # дельта вес
            synopsis[1] += deltaWeight  # обновление веса
            synopsis[2] = deltaWeight  # обновление дельта веса

    def activate(self, val=None):
        if not val:
            val = self.value
        self.value = 1 / (1 + exp(0 - val))  # Сигмоид

    def calculateValue(self):
        neighborSum = 0
        for i in self.backNeighbors:
            neighborSum += i[0].value * i[1]
        return neighborSum
