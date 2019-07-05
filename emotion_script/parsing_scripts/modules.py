import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2


def get_iou(box_a, box_b):
    """
    Calculate intersection over union metric for two boxes
    :param box_a: coordinates of box A
    :param box_b: coordinates of box B
    :return: intersection over union
    """

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection_area = abs(x_b - x_a + 1) * abs(y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
    return iou


def plot_boxes(path_to_parsed_file):
    """
    Plot images with face bounding boxes
    :param path_to_parsed_file: path to file with parsed data
    """

    # Max number of images is 25
    num_rows, num_cols = 5, 5
    figure_size = [10, 10]
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size)
    with open(path_to_parsed_file, 'r') as f:
        file = [s.split(' ') for s in f.read().split('\n')]
        try:
            for i, axi in enumerate(ax.flat):
                image = np.array(Image.open(file[i][0]))

                # Colour of man/woman box will be blue/red
                if int(file[i][1]) == 0:
                    box = patches.Rectangle((float(file[i][2]), float(file[i][3])), float(file[i][4]),
                                            float(file[i][5]), linewidth=3, edgecolor='b', facecolor='none')
                else:
                    box = patches.Rectangle((float(file[i][2]), float(file[i][3])), float(file[i][4]),
                                            float(file[i][5]), linewidth=3, edgecolor='r', facecolor='none')
                axi.imshow(image)
                axi.add_patch(box)
        except Exception as e:
            print(e)
        plt.show()


def extract_and_order_train_data(mnist_data):
    data = mnist_data['training_images']
    labels = mnist_data['training_labels']
    assert data.shape[0] == 60000
    order = np.argsort(labels)
    return data[order]


def draw_and_show_image_by_counter(counter, data):
    image = data[counter]

    image = cv2.resize(image, (0, 0), fx=30, fy=30,
                       interpolation=cv2.INTER_NEAREST)

    image = cv2.copyMakeBorder(image, 100, 0, 0, 0,
                               borderType=cv2.BORDER_CONSTANT, value=60)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{counter}'
    cv2.putText(image, text, (10, 50), font, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('data', image)


def mnist_train_visualization_image_per_windows(mnist_data):

    data = extract_and_order_train_data(mnist_data)

    counter, max_counter = 0, 60000

    while True:

        draw_and_show_image_by_counter(counter, data)

        key = cv2.waitKeyEx(100)
        if key == 2555904:  # right arrow key
            counter += 1
            counter %= max_counter
        elif key == 2424832:  # left arrow key
            counter -= 1
            counter %= max_counter
        elif key == 102:  # f (forward) to incease by 1000
            counter += 1000
            counter %= max_counter
        elif key == 98:  # b (backward) to incease by 1000
            counter += 1000
            counter %= max_counter
        elif key == 113:
            break

    cv2.destroyAllWindows()
