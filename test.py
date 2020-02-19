# filename = './data/iris/iris.trn'
# dataset = load_csv(filename)
# for i in range(len(dataset[0])-1):
#     str_column_to_float(dataset, i)
# print(dataset)
# # convert class column to integers
# str_column_to_int(dataset, len(dataset[0])-1)
# # evaluate algorithm
# n_folds = 5
# num_neighbors = 3
# scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
from math import sqrt
from csv import reader
from decimal import *
getcontext().prec = 3000
Decimal.lim
# def load_file(file_name):
# 	data_set = list()
# 	with open(file_name,'r') as file:
# 		csv_reader = reader(file)
# 		for row in csv_reader:
# 			if not row:
# 				continue
# 			data_set.append(row)
# 	return data_set

# def str_to_float(data_set, col):
# 	for row in data_set:
# 		row[col] = float(row[col].strip())


# def str_to_int(data_set, col):
#     # class_values = [row[column] for row in dataset]
#     # unique = set(class_values)
#     # lookup = dict()
#     # for i, value in enumerate(unique):
#     #     lookup[value] = i
#     # for row in dataset:
#     #     row[column] = lookup[row[column]]
#     # return lookup
#     for row in data_set:
#     	row[col] = int(row[col].strip())

# def minkowski_distance(test_row, train_row, q):
# 	distance = Decimal()
# 	for i in range(len(train_row)-1):
# 		temp = Decimal((test_row[i]- train_row[i]))
# 		distance += temp**Decimal(q)
# 	value = distance**Decimal(1/q)
# 	return value

# train_file_path = './data/fp/fp.trn'
# test_file_path = './data/fp/fp.tst'
# data_set = load_file(train_file_path)
# test_set = load_file(test_file_path)

# for i in range(len(data_set[0])-1):
# 	str_to_float(data_set, i)

# for i in range(len(test_set[0])-1):
# 	str_to_float(test_set, i)


# str_to_int(data_set, len(data_set[0])-1)

# str_to_int(test_set, len(test_set[0])-1)
# dim = len(data_set[0])-1

# d = minkowski_distance(test_set[0],data_set[0], dim)
# print(d)