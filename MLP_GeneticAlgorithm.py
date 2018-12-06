import numpy
import random
import math
import copy

class MLP_GeneticAlgorithm:

    __generation = 0
    __individual = []  # test
    __num_individual = 0
    __num_chromosome = 0
    __num_generation = 0
    __mlp_structure = []
    __report = None

    def __init__(self, num_generation, num_individual):
        self.__num_individual = num_individual
        self.__num_generation = num_generation
        self.__report = open("ConfusionMatrixReport.txt", "a")
        return

    def init_mlp(self, arr_structure, rang_of_population):
        self.__mlp_structure = arr_structure
        self.__num_chromosome = arr_structure[0]
        for i in range(1, len(arr_structure)):
            self.__num_chromosome += arr_structure[i] + (arr_structure[i] * arr_structure[i - 1])
        # init population
        for i in range(self.__num_individual):
            arr = []
            for j in range(self.__num_chromosome):
                if j < arr_structure[0]:
                    arr.append(0)
                else:
                    arr.append(random.uniform(-rang_of_population, rang_of_population))
            self.__individual.append(arr)
        return

    def generate(self, arr_data, arr_design_output):  # number of arr_data must equal number of arr_design_output
        t = 0
        result = []
        min_fitness = 99999
        while t <= self.__generation:
            # calculate fitness
            fitness = []
            for chromosome in self.__individual:
                error = 0
                for index in range(len(arr_data)):
                    output = self.feed_forward(arr_data[index], chromosome)
                    error += self.fitness_error(output, arr_design_output[index])
                fitness.append(error / len(arr_data))

            if min(fitness) < min_fitness:
                result = copy.deepcopy(self.__individual[fitness.index(min(fitness))])
                min_fitness = min(fitness)

            pool_fitness = []
            mating_pool = []
            child_pool = []

            # selection parent
            # guarantee parent one
            mating_pool.append(self.__individual.pop(fitness.index(min(fitness))))
            pool_fitness.append(fitness.pop(fitness.index(min(fitness))))
            for i in range(0, int(len(self.__individual) * 0.1) - 1):  # keep chromosome has best fitness
                mating_pool.append(self.__individual.pop(fitness.index(min(fitness))))
                pool_fitness.append(fitness.pop(fitness.index(min(fitness))))

            # cross over
            while True:
                if len(mating_pool) + len(child_pool) + 2 > int(self.__num_individual*0.9):
                    break
                # type 2: best fitness cross best fitness
                index_mating1 = random.randint(0, len(mating_pool) - 1)
                index_mating2 = random.randint(0, len(mating_pool) - 1)
                x, y = self.cross_over(mating_pool[index_mating1], self.__individual[index_mating2])
                child_pool.append(x)
                child_pool.append(y)

            # elitism
            while len(mating_pool) + len(child_pool) < self.__num_individual:
                index = random.randint(0, len(self.__individual) - 1)
                mating_pool.append(self.__individual[index])
                pool_fitness.append(fitness[index])

            # mutation
            for index_chromosome in range(len(mating_pool)):
                self.fitness_mutation(mating_pool[index_chromosome], pool_fitness[index_chromosome], 0.1)

            mating_pool.extend(child_pool)
            # to next generation
            self.__individual = mating_pool
            t += 1
        return result

    def feed_forward(self, arr_input, chromosome):
        # output of each layer
        input_of_layer = arr_input

        # point index
        weight_index = self.__mlp_structure[0]
        bias_index = self.__mlp_structure[0]

        for layer_index in range(1, len(self.__mlp_structure)):  # each layer
            arr = []
            bias_index += self.__mlp_structure[layer_index] * self.__mlp_structure[layer_index - 1]

            for node_index in range(self.__mlp_structure[layer_index]):  # each node
                v = 0
                for input_index in range(self.__mlp_structure[layer_index - 1]):  # each wire. number of wire must equal number of input
                    i_index = input_index
                    w_index = weight_index + (self.__mlp_structure[layer_index] * input_index)
                    v += chromosome[w_index] * input_of_layer[i_index]
                b_index = bias_index
                v += chromosome[b_index]
                arr.append(self.activation_func(v))
                # update index
                bias_index += 1
                weight_index += 1
            weight_index = bias_index
            input_of_layer = arr

        return input_of_layer

    # sigmoid function
    @staticmethod
    def activation_func(v):
        y = 1 / (1 + math.exp(-v))
        return y

    @staticmethod
    def fitness_error(output, design_output):
        error_output = numpy.subtract(design_output, output)
        e = numpy.sum(numpy.fabs(error_output)) / len(error_output)
        return e

    @staticmethod
    def cross_over(parent1, parent2):
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2)
        for i in range(len(p1)):
            pc = random.uniform(0, 1)
            if pc > 0.5:
                temp = p1[i]
                p1[i] = p2[i]
                p2[i] = temp
        return p1, p2

    @staticmethod
    def strong_mutation(chromosomes, learning_rate):
        for chromosome in chromosomes:
            p = random.uniform(0, 1)
            if p < 0.5:
                chromosome += random.uniform(-learning_rate, learning_rate)
        return

    @staticmethod
    def fitness_mutation(chromosomes, fitness, learning_rate):
        for chromosome in chromosomes:
            p = random.uniform(0, 1)
            if p < fitness:
                chromosome += random.uniform(-learning_rate, learning_rate)
        return

    # test neuron network
    @staticmethod
    def init_array(num_arr):
        return [0] * num_arr

    def test_classification(self, test_arr, design_arr, chromosome, round_test):
        result = 0
        confusion_matrix = []
        for i in range(len(design_arr[0])):
            confusion_matrix.append(self.init_array(len(design_arr[0])))  # design output 1 or 0

        self.__report.write("fold {}\n".format(round_test))
        self.__report.write("chromosome: {}\n".format(chromosome))
        for cur_row in range(len(test_arr)):
            # feed data
            output = self.feed_forward(test_arr[cur_row], chromosome)
            # set 1 0 and output compare design output
            assume_index = output.index(max(output))
            # set 1 0 output
            for index_output in range(len(output)):
                if index_output == assume_index:
                    output[index_output] = 1
                else:
                    output[index_output] = 0

            # update matrix
            confusion_matrix[assume_index][design_arr[cur_row].index(max(design_arr[cur_row]))] += 1
            if numpy.array_equal(design_arr[cur_row], output):
                result += 1

        result = round(result * 100 / len(test_arr), 2)
        self.__report.write("{}\n".format(result))
        for i in range(len(confusion_matrix)):
            self.__report.write("{}\n".format(confusion_matrix[i]))
        self.__report.write("\n")
        return
