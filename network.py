from random import uniform
from math import exp
from typing import List
import numba


class Network:
    def __init__(self, shape, learningRate=0.5, momentum=0.25):

        self.generations = 0
        self.iterations = 0
        self.momentum = momentum
        self.learningRate = learningRate

        self.weights = self.generateWeights(shape)
        self.shape = shape

    @staticmethod
    def activateInputs(inputs: List[float]) -> List[float]:
        return list(map(Network.activate, inputs))

    def train(self, dataset):
        pass
        self.weights: List[List[List[float]]]
        for _ in range(1):
            for data in dataset:
                result = self.calculateLayers(data[0], self.weights)
                self.weights = self.trainItteration(result, data[1], self.weights)
                self.iterations += 1
            self.generations += 1

    @staticmethod
    def trainItteration(result: List[List[float]], actual: List[float], weights):
        # TODO: write implementation
        pass


    @staticmethod
    def getError(data=-1):
        return data

    def calculate(self, inputs: List[float]):
        """
        Network.calculateLayers wrapper
        """
        return self.calculateLayers(inputs, self.weights)

    @staticmethod
    def calculateLayers(inputs: List[float], weights: List[List[List[float]]]) -> List[List[float]]:
        outputLayers = [Network.activateInputs(inputs)]
        for weightsLayer in weights:
            outputLayers.append(Network.calculateSingleLayer(outputLayers[-1], weightsLayer))
        return outputLayers

    @staticmethod
    # @numba.jit(cache=True, fastmath=True, nopython=True)
    # sum() unsupported with numba
    # TODO: rewrite without sum()
    def calculateSingleLayer(inputLayer: List[float], weightsLayer: List[List[float]], bias=1.0) -> List[float]:
        # weightsLayer: [*neuronsWeights, biasWeight]
        inputLayer = inputLayer.copy()
        inputLayer.append(bias)
        return [
            Network.activate(
                sum(
                    inputLayer[connection] * outputNeuron[connection]
                    for connection in range(len(outputNeuron))
                )
            )
            for outputNeuron in weightsLayer
        ]

    @staticmethod
    @numba.jit(cache=True, nopython=True, fastmath=True)
    def activate(value: float) -> float:
        return 1 / (1 + exp(-value))  # Sigmoid activation

    @staticmethod
    def derivative(value: float) -> float:
        return (1 - value) * value  # Sigmoid derivative

    @staticmethod
    def generateWeights(structure: List[int]) -> List[List[List[float]]]:
        return [
            [
                [uniform(-1, 1) for _ in range(structure[connectionLayer] + 1)]
                for _ in range(structure[connectionLayer+1])
            ]
            for connectionLayer in range(len(structure) - 1)
        ]
