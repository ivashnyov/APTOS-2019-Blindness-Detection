import numpy as np
from tools.create_dataset import create


def split(files, train_size=1):
    """
    Split all data on train and val
    :param files: list of paths to files with parsed data
    :param train_size: size of train part from 0 to 1. Size of val part is 1 - train_size
    """

    # Combine all parsed files together
    labels = create(files)

    # Shuffle lines
    labels = np.array(labels)
    np.random.shuffle(labels)

    # Calculate the number of train and val examples
    num_trains = int(train_size * len(labels))
    num_vals = len(labels) - int(train_size * len(labels))

    # Write train file
    with open('../data/labels_affer_train_emotion.txt', 'w') as f:
        for label in labels[:num_trains]:
            out_string = ' '.join(label) + '\n'
            f.write(out_string)

    # Write val file
    with open('../data/labels_affer_test_emotion.txt', 'w') as f:
        for label in labels[-num_vals:]:
            out_string = ' '.join(label) + '\n'
            f.write(out_string)

    print("From %d elements was created train file with %d elements and val file with %d elements" % (len(labels),
                                                                                                      num_trains,
                                                                                                      num_vals))


if __name__ == '__main__':

    split(['../data/labels_fer_emotion.txt', '../data/labels_affectnet_manual_emotion.txt'])
