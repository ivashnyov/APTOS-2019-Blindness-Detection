import cv2
import numpy as np
import torch
import traceback
from datetime import datetime
from sklearn.metrics import confusion_matrix

from tools.squeezenet import squeezenet1_1
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import time
import os
from tools.create_dataset import create
from tools.train_val_split import split

from imgaug import augmenters as iaa
from torchvision import transforms


class EmotionDataset(Dataset):
    def __init__(self, labels_files, pad=0.4, image_size=(244, 244), transforms=None):
        self.image_size = image_size
        self.pad = pad
        self.dataset = np.array(create(labels_files))
        self.transforms = transforms

    @staticmethod
    def get_padded_bbox(x1, y1, w, h, image_size, pad):
        """
        Convert box coordinates and padding them
        :param x1: x coordinate of low left corner
        :param y1: y coordinate of low left corner
        :param w: width of box
        :param h: height of box
        :param image_size: image size
        :param pad: scale of padding
        :return: new coordinates
        """

        x2, y2 = x1 + w, y1 + h
        x1 = max(x1 - pad * w, 0)
        y1 = max(y1 - pad * h, 0)
        x2 = min(x2 + pad * w, image_size[1])
        y2 = min(y2 + pad * h, image_size[0])
        return int(x1), int(y1), int(x2), int(y2)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = self.dataset[index]
        try:
            path, emotion, x1, y1, w, h = example
        except Exception as e:
            print(e)
            print(example)
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)

        emotion_set = np.zeros(7, dtype=np.float)
        emotion_set[int(emotion)] = 1

        image = cv2.imread(path)[..., :: -1]
        x1, y1, x2, y2 = self.get_padded_bbox(x1, y1, w, h, image.shape[:2], self.pad)

        image_cropped = cv2.resize(image[y1: y2, x1: x2], self.image_size)
        if self.transforms:
            image_cropped = self.transforms(image_cropped)
        image_cropped = image_cropped / 255.
        image_cropped = image_cropped.transpose(2, 0, 1)
        return image_cropped, emotion_set


def calculate_weights(labels):
    """
    Calculate weights of classes in dataset
    :param labels: list of files with parsed data
    :return: weights of classes
    """

    classes = {'happy': 0, 'sad': 0, 'neutral': 0, 'other': 0}
    seq_type = []
    seq_weight = []
    dataset = create(labels)
    for example in dataset:
        emotion = example[1]
        if emotion == '3':
            classes['happy'] += 1
            seq_type.append('happy')
        elif emotion == '4':
            classes['sad'] += 1
            seq_type.append('sad')
        elif emotion == '6':
            classes['neutral'] += 1
            seq_type.append('neutral')
        else:
            classes['other'] += 1
            seq_type.append('other')

    for i in classes.keys():
        classes[i] = 1 / classes[i]

    for type_img in seq_type:
        seq_weight.append(classes[type_img])

    return seq_weight


def evaluate(model, batch_size):
    emotions = {0: 3, 1: 3, 2: 3, 3: 0, 4: 1, 5: 3, 6: 2}
    val_dataset = EmotionDataset(VAL_DATASET_PATH)
    sampler_val = WeightedRandomSampler(weights=calculate_weights(VAL_DATASET_PATH), num_samples=len(val_dataset))
    val_bg = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val)
    model = model.eval()
    # Loss per batch on validation data
    loss_emotion_mean = 0.0
    bg_iter = iter(val_bg)
    acc = 0.0
    pred = [0] * 7
    true = [0] * 7
    for step in range(int(len(val_dataset) / batch_size)):
        image_batch, emotion_batch = next(bg_iter)
        image_batch = image_batch.float().cuda()
        emotion_batch = emotion_batch.float().cuda()
        out = model(image_batch)
        loss_emotion = nn.CrossEntropyLoss()(out, torch.max(emotion_batch, 1)[1])
        loss_emotion_mean += loss_emotion.cpu().data.numpy().item()
        predictions = torch.argmax(out[:, :7], dim=1).tolist()
        answers = torch.argmax(emotion_batch, dim=1).tolist()
        for i in range(len(answers)):
            answers[i] = emotions[answers[i]]
            predictions[i] = emotions[predictions[i]]
        pred.extend(predictions)
        true.extend(answers)
        acc += torch.sum(torch.tensor(predictions) == torch.tensor(answers)).float() / float(
            batch_size)

    loss_emotion_mean /= int(len(val_dataset) / batch_size)
    acc /= int(len(val_dataset) / batch_size)
    matrix = confusion_matrix(true, pred)

    return loss_emotion_mean, acc.item(), matrix


if __name__ == '__main__':
    # Create CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    VAL_DATASET_PATH = ['data/labels_cohn_kanade_emotion.txt']
    classifier = squeezenet1_1(pretrained=False, num_classes=1000)
    classifier.classifier[1] = nn.Conv2d(512, 7, kernel_size=1)
    classifier.num_classes = 7
    checkpoints = os.listdir('checkpoints/aff+fer7/')
    for checkpoint in checkpoints:
        classifier.load_state_dict(torch.load('checkpoints/aff+fer7/' + checkpoint,
                                              map_location='cpu'))
        classifier.eval()
        if torch.cuda.is_available():
            classifier.cuda()
        loss, acc, matrix = evaluate(classifier, 64)
        print(loss, acc)
        print(matrix)
        name = checkpoint.split('_')
        print(name[0] + '_' + name[1] + '_' + name[2] + '_' + f'{loss:.5f}' + '_' + name[4] + '_' + f'{acc:.5f}'
                  + '.pth')
