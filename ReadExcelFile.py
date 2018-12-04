import xlrd
import random
import math
import numpy
import copy

class ReadExcelFile:
    table_data = []
    fold_data = []
    num_row = 0
    num_col = 0

    min = 0
    max = 0
    j = 0
    parameter = 0
    def __init__(self, file_name, parameter):
        self.parameter = parameter
        # read excel file
        workbook = xlrd.open_workbook(file_name)
        worksheet = workbook.sheets()

        self.num_row = worksheet[0].nrows
        self.num_col = worksheet[0].ncols

        for cur_row in range(1, self.num_row):
            row = []
            for cur_col in range(0, self.num_col):
                value = worksheet[0].cell_value(cur_row, cur_col)
                row.append(value)
            self.table_data.append(row)

        # find statistics value
        self.max = numpy.max(self.table_data)
        self.min = numpy.min(self.table_data)

        temp = self.max
        while True:
            temp = temp / 10
            self.j += 1
            if temp < 1:
                break

        # random data
        for i in range(0, len(self.table_data)):
            rand = random.randrange(0, len(self.table_data) - 1)
            temp = self.table_data[i]
            self.table_data[i] = self.table_data[rand]
            self.table_data[rand] = temp
        return

    def ten_fold_data(self):
        # cut to test set
        num_of_train = math.ceil((len(self.table_data) * 10) / 100)

        for i in range(0, 9):
            temp_arr = []
            for j in range(0, num_of_train):
                temp_arr.append(self.table_data.pop(random.randrange(0, len(self.table_data) - 1)))
            self.fold_data.append(temp_arr)
        self.fold_data.append(self.table_data)
        return

    def minmax_normalize(self, new_min, new_max):
        for cur_row in range(0, len(self.table_data)):
            for cur_col in range(0, len(self.table_data[cur_row])):
                temp = (self.table_data[cur_row][cur_col] - self.min) / (self.max - self.min) * (new_max - new_min)
                self.table_data[cur_row][cur_col] = temp + new_min
        return

    def z_score(self):
        mean_arr = []
        sd_arr = []
        temp_arr = copy.deepcopy(self.table_data)
        temp_arr = numpy.matrix(temp_arr)
        for cur_col in range(0, self.parameter):
            mean_arr.append(numpy.mean(temp_arr[:, cur_col]))
            sd_arr.append(numpy.std(temp_arr[:, cur_col]))

        for cur_row in range(0, len(self.table_data)):
            for cur_col in range(0, self.parameter):
                self.table_data[cur_row][cur_col] = (self.table_data[cur_row][cur_col] - mean_arr[cur_col])/sd_arr[cur_col]
        return

    def decimal_scaling(self):
        for cur_row in range(0, len(self.table_data)):
            for cur_col in range(0, len(self.table_data[cur_row])):
                self.table_data[cur_row][cur_col] = self.table_data[cur_row][cur_col]/math.pow(10, self.j)
        return

    def split_output(self, arr_data):
        input_arr = []
        output_arr = []
        for i in range(len(arr_data)):
            temp_input = []
            temp_output = []
            for j in range(len(arr_data[i])):
                if j < self.parameter:
                    temp_input.append(arr_data[i][j])
                else:
                    temp_output.append(arr_data[i][j])
            input_arr.append(temp_input)
            output_arr.append(temp_output)
        return input_arr, output_arr
