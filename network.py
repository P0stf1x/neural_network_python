from neuron import Neuron
import numpy as np


class Network:
    def __init__(self, layerStructure, learningRate=0.5, momentum=0.25):
        self.layers = []
        self.create(layerStructure)
        self.bias = Neuron(value=1)  # Нейрон смещения
        self.passNeighbors()
        self.generations = 0
        self.iterations = 0
        self.learningRate = learningRate
        self.momentum = momentum
        self.error = 0

    def create(self, layerStructure):
        for i in layerStructure:
            temp = [Neuron() for _ in range(i)]  # Скрытые, выходной слои
            self.layers.append(temp)

    def getValues(self):
        temp = []
        for i in self.layers[-1]:
            temp.append(i.value)
        return temp

    # @numba.njit(fastmath=True, cache=True)
    def calculate(self, inputs):
        for neuron in range(len(self.layers[0])):
            self.layers[0][neuron].value = inputs[neuron]
        for layer in range(1, len(self.layers)):
            for neuron in self.layers[layer]:
                neuron.activate(neuron.calculateValue())
        return self.getValues()

    def passNeighbors(self):
        for j in range(1, len(self.layers)):
            for i in self.layers[j]:
                i.initNeighbors(self.layers[j - 1].__add__([self.bias]))

    def getError(self, sets):
        error = 0
        if len(sets) > 0:
            for i, dataset in enumerate(sets):
                self.calculate(dataset[: len(self.layers[0])])
                for j in range(len(self.getValues())):
                    error += (
                        dataset[: np.negative(len(self.layers[0])) : -1][j]
                        - self.getValues()[j]
                    ) ** 2
            error /= len(sets)
        return error

    def iteration(self, dataset):
        # Считаем дельту для каждого нейрона
        self.calculate(dataset[: len(self.layers[0])])
        for i, neuron in enumerate(self.layers[-1]):
            pr = (
                1 - neuron.value
            ) * neuron.value  # Производная функции активации нейрона (сигмоида)
            neuron.delta = (
                dataset[: np.negative(len(self.layers[0])) : -1][i] - neuron.value
            ) * pr
        for layerid in range(len(self.layers) - 2, 0, -1):
            for neuronid, neuron in enumerate(self.layers[layerid]):
                deltaSum = 0
                for nextNeuron in self.layers[layerid + 1]:
                    deltaSum += nextNeuron.backNeighbors[neuronid][1] * nextNeuron.delta
                pr = (1 - neuron.value) * neuron.value
                neuron.delta = pr * deltaSum

        # Обновляем веса нейронов
        for layer in self.layers[:0:-1]:
            for neuron in layer:
                neuron.updateWeight(self.learningRate, self.momentum)

    def train(self, sets):
        for _ in range(100):  # Для ускорения обучения проходим 100 поколений за раз
            for dataset in sets:
                self.iteration(dataset)
                self.iterations += 1
            self.generations += 1
        self.error = self.getError(sets)
