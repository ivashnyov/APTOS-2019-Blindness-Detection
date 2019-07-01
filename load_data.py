import csv
import numpy as np


def split_train_test(path_to_file, train_test_ratio, save):
    data = []
    with open(path_to_file, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            data.append(line)

        np.random.shuffle(data)
        num_train = train_test_ratio * len(data)

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


if __name__ == '__main__':
    split_train_test('W:/APTOS 2019 Blindness Detection/train.csv')