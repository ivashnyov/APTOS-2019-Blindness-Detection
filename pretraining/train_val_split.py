import csv
import numpy as np


def split_train_test_new(path_to_file, train_test_ratio, save):
    data = []
    with open(path_to_file, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            data.append(line)

        np.random.shuffle(data)
        num_train = int(train_test_ratio * len(data))

        train = []
        test = []

        for i in range(num_train):
            train.append(data[i])

        for i in range(num_train, len(data)):
            test.append(data[i])

        if save:
            np.save('train', train)
            np.save('test', test)

        return train, test


def split_train_test_old(path_to_file_test, path_to_file_train):
    data = []
    with open(path_to_file_train, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            data.append([line[0], line[1]])

    with open(path_to_file_test, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            data.append([line[0], line[1]])

    return data


if __name__ == '__main__':
    split_train_test_new('W:/APTOS 2019 Blindness Detection/train.csv')
