# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
 
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
# def dataset_minmax(dataset):
#     minmax = list()
#     for i in range(len(dataset[0])):
#         col_values = [row[i] for row in dataset]
#         value_min = min(col_values)
#         value_max = max(col_values)
#         minmax.append([value_min, value_max])
#     return minmax
 
# # Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, minmax):
#     for row in dataset:
#         for i in range(len(row)):
#             row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

#calculate euclidean distance
def euclidean_distance(rtest, rdata):
    distance = 0.0
    for i in range(len(rdata)-1):
        distance += (rtest[i] - rdata[i])**2
    return sqrt(distance)

#get similar neighbors
def get_neighbors(data_set, test_row, num_neighbors):
    distances = list()
    for r in data_set:
        d = euclidean_distance(test_row, r)
        distances.append((r,d))
    distances.sort(key=lambda tmp: tmp[1])
    neighbors = list()
    for i in range(num_neighbors):
        #print("Neighbor: ", distances[i][0])
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return (predictions)

data_set= [
    [ 0.376000, 0.488000, 0],
    [ 0.312000, 0.544000, 0],
    [ 0.298000, 0.624000, 0],
    [ 0.394000, 0.600000, 0],
    [ 0.506000, 0.512000, 0],
    [ 0.488000, 0.334000, 1],
    [ 0.478000, 0.398000, 1],
    [ 0.606000, 0.366000, 1],
    [ 0.428000, 0.294000, 1],
    [ 0.542000, 0.252000, 1],
]


test_set = [
    [ 0.550000, 0.364000],
    [ 0.558000, 0.470000],
    [ 0.456000, 0.450000],
    [ 0.450000, 0.570000],
]

print(k_nearest_neighbors(data_set,test_set, 1))

filename = './data/iris/iris.trn'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
print(dataset)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
num_neighbors = 3
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))