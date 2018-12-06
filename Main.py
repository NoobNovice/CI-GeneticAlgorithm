from MLP_GeneticAlgorithm import MLP_GeneticAlgorithm
from ReadExcelFile import ReadExcelFile
import tqdm
import copy
import numpy

data = ReadExcelFile("wdbc.xls", 30)
data.z_score()
data.ten_fold_data()

for i in range(0, 10):
    # prepare data
    fold_data = copy.deepcopy(data.fold_data)
    test_data = fold_data.pop(i)
    train_data = numpy.concatenate(fold_data)
    # init module
    data_input, design_output = data.split_output(train_data)
    ga = MLP_GeneticAlgorithm(900, 50)
    ga.init_mlp([30, 15, 7, 2], 3)
    # test module
    module = ga.generate(data_input, design_output)
    print(module)
    test_input, test_output = data.split_output(test_data)
    ga.test_classification(test_input, test_output, module, i + 1)
