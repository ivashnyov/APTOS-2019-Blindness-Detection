import os
import platform
import traceback
import cv2
import numpy as np
import pandas as pd
from mtcnn import mtcnn
from tqdm import tqdm
from modules import get_iou, plot_boxes


def parser(dataset_path, num_images=None):
    mtcnn_model = mtcnn.MTCNN()

    cascade_path = '../face_two'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Create list of labels and path to images
    labels = []
    with open(f'{dataset_path}/Annotations/Fer2013.txt', 'r') as f:
        file = [s.split(',') for s in f.read().split('\n')]
        for l in file[1:]:
            if l[0]:
                path = f'Images/{l[0]}.jpg'
                for i in range(1, len(l)):
                    if l[i] == '1':
                        emotion = i - 1
                        break
                labels.append([path, emotion])

    # How many images will be parsed
    if num_images is not None:
        labels = labels[:num_images]

    # For each file find face bounding box and store all together
    for label in tqdm(labels):
        try:
            image_path, gender = label[0], label[1]
            image = cv2.imread(f'{dataset_path}/{image_path}')
            mtcnn_bbox = mtcnn_model.detect_faces(image)

            # if mtcnn has found face on the image
            if len(mtcnn_bbox) != 0:
                # Enlarge bounding box that was found by mtcnn
                s = [d['box'][2] * d['box'][3] for d in mtcnn_bbox]
                max_size_ind = np.argmax(s)
                x1, y1, w, h = mtcnn_bbox[max_size_ind]['box']
                x2, y2 = x1 + w, y1 + h
                x1 = int(max(x1 - 0.4 * w, 0))
                y1 = int(max(y1 - 0.4 * h, 0))
                x2 = int(min(x2 + 0.4 * w, image.shape[1]))
                y2 = int(min(y2 + 0.4 * h, image.shape[0]))

                # Try to find face in this enlarged box
                cascade_bbox = face_cascade.detectMultiScale(image[y1:y2, x1:x2])

                # If face was found
                if len(cascade_bbox) != 0:
                    s = [d[2] * d[3] for d in cascade_bbox]
                    max_size_ind = np.argmax(s)
                    x1_c, y1_c, x2_c, y2_c = cascade_bbox[max_size_ind][0], cascade_bbox[max_size_ind][1], \
                                             cascade_bbox[max_size_ind][0] + cascade_bbox[max_size_ind][2], \
                                             cascade_bbox[max_size_ind][1] + cascade_bbox[max_size_ind][3]

                    # Calculate IoU for two boxes
                    iou = get_iou([x1, y1, x2, y2], [x1 + x1_c, y1 + y1_c, x1 + x2_c, y1 + y2_c])
                    # Check if boxes overlap each other enough than stores all together
                    if iou > 0.25:
                        with open('../data/labels_fer_emotion.txt', 'a') as f:
                            out_string = f'{dataset_path}/{image_path} {gender} ' \
                                f'{x1 + x1_c} {y1 + y1_c} {x2_c - x1_c} {y2_c - y1_c}\n'
                            f.write(out_string)
        except Exception as e:
            print(traceback.format_exc())


if __name__ == '__main__':
    # Create CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Check platform type
    if platform.system() == 'Windows':
        # Set "datasets" disk name
        datasets_path = 'W:/'
    else:
        datasets_path = '/mnt/dataserver/inbox/'

    dataset_path = os.path.join(datasets_path, '_for_ilya/DATASETS/Fer2013')
    # Start script
    parser(dataset_path=dataset_path)
