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
import torch.utils.model_zoo as model_zoo
from tools.deepemotion import DeepEmotion

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


def log(message, message_type='INFO'):
    """
    Log training info into file
    :param message: message from training function
    :param message_type: type of this message
    """

    print(f'{message_type}: {datetime.now()}: {message}')
    with open(LOG_FILE, 'a') as f:
        f.write(f'{message_type}: {datetime.now()}: {message}\n')


class Contrast(object):
    def __init__(self):
        self.hue = iaa.ContrastNormalization((0.1, 3))

    def __call__(self, image):
        image_aug = self.hue.augment_image(image)
        return image_aug


class Brightness(object):
    def __init__(self):
        self.hue = iaa.Add((-30, 30))

    def __call__(self, image):
        image_aug = self.hue.augment_image(image)
        return image_aug


# Class for affine augmentation (rotation)
class Affine(object):
    def __init__(self, rotate=(-25, 25)):
        self.rotate = iaa.Affine(rotate=rotate)

    def __call__(self, image):
        image_aug = self.rotate.augment_image(image)
        return image_aug


# Class for horizontal flip augmentation
class Flip(object):
    def __init__(self, t=0.5):
        self.flip = iaa.Fliplr(t)

    def __call__(self, image):
        image_aug = self.flip.augment_image(image)
        return image_aug


# Class for piecewise affine augmentation
class PiecewiseAffine(object):
    def __init__(self, scale=(0.01, 0.02)):
        self.PiecewiseAffine = iaa.PiecewiseAffine(scale)

    def __call__(self, image):
        image_aug = self.PiecewiseAffine.augment_image(image)
        return image_aug


# Class for blur augmentation
class AverageBlur(object):
    def __init__(self, k):
        self.AverageBlur = iaa.AverageBlur(k)

    def __call__(self, image):
        image_aug = self.AverageBlur.augment_image(image)
        return image_aug


# Class for random patch augmentation
class RandomPatch(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image_aug = image
        xs = np.random.randint(image_aug.shape[1], size=8)
        ys = np.random.randint(image_aug.shape[0], size=8)
        ws = np.random.randint(60, size=8)
        hs = np.random.randint(60, size=8)
        cs = np.random.randint(255, size=8)

        for x, y, w, h, c in zip(xs, ys, ws, hs, cs):
            cv2.rectangle(image_aug, (x, y), (x + w, y + h), (int(c), int(c), int(c)), -1)

        return image_aug


# Convert dataset file into proper form for training
class EmotionDataset(Dataset):
    def __init__(self, labels_files, pad=0.4, image_size=(244, 244), transforms=None, filters=False):
        self.image_size = image_size
        self.pad = pad
        self.dataset = np.array(create(labels_files))
        self.transforms = transforms
        self.filters = filters

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

        emotion_set = np.zeros(7, dtype=np.long)
        emotion_set[int(emotion)] = 1

        image = cv2.imread(path)[..., :: -1]
        x1, y1, x2, y2 = self.get_padded_bbox(x1, y1, w, h, image.shape[:2], self.pad)

        image_cropped = cv2.resize(image[y1: y2, x1: x2], self.image_size)
        if self.transforms:
            image_cropped = self.transforms(image_cropped)
        if self.filters:
            im = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(im, cv2.CV_64F)
            sobel = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=5)
            r_channel, g_channel, b_channel = cv2.split(image_cropped)
            image_cropped = np.dstack((r_channel, g_channel, b_channel, sobel, laplacian))

        image_cropped = image_cropped / 255.
        image_cropped = image_cropped.transpose(2, 0, 1)
        return image_cropped, emotion_set


def calculate_weights(labels):
    """
    Calculate weights of classes in dataset
    :param labels: list of files with parsed data
    :return: weights of classes
    """

    classes = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    seq_type = []
    seq_weight = []
    dataset = create(labels)
    for example in dataset:
        emotion = example[1]
        if emotion == '0':
            classes['angry'] += 1
            seq_type.append('angry')
        elif emotion == '1':
            classes['disgust'] += 1
            seq_type.append('disgust')
        elif emotion == '2':
            classes['fear'] += 1
            seq_type.append('fear')
        elif emotion == '3':
            classes['happy'] += 1
            seq_type.append('happy')
        elif emotion == '4':
            classes['sad'] += 1
            seq_type.append('sad')
        elif emotion == '5':
            classes['surprise'] += 1
            seq_type.append('surprise')
        elif emotion == '6':
            classes['neutral'] += 1
            seq_type.append('neutral')

    for i in classes.keys():
        classes[i] = 1 / classes[i]

    for type_img in seq_type:
        seq_weight.append(classes[type_img])

    return seq_weight


def fast_gradient_sign_method(model, X, y, epsilon=0.1):
    """
    Construct FGSM adversarial examples on the examples X
    :param model: model
    :param X: batch
    :param y: answers
    :param epsilon: coefficient
    :return: adversarial noise
    """

    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """
    Construct FGSM adversarial examples on the examples X
    :param model: model
    :param X: batch
    :param y: answers
    :param epsilon: max values
    :param alpha: coefficient per iteration
    :param num_iter: number of iterations
    :param randomize: randomized or zeros starting pixels
    :return: adversarial noise
    """

    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = F.binary_cross_entropy_with_logits(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def train(model, batch_size, num_epochs, adversarial_training=False):
    """
    Training model
    :param model: model to train
    :param batch_size: size of batch
    :param num_epochs: number of epochs
    :param adversarial_training: use or not adversarial training
    """

    # Create train dataset with augmentation
    train_dataset = EmotionDataset(TRAIN_DATASET_PATH)
    sampler_train = WeightedRandomSampler(weights=calculate_weights(TRAIN_DATASET_PATH), num_samples=len(train_dataset))
    train_bg = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train)
    # Create validation dataset
    val_dataset = EmotionDataset(VAL_DATASET_PATH)
    sampler_val = WeightedRandomSampler(weights=calculate_weights(VAL_DATASET_PATH), num_samples=len(val_dataset))
    val_bg = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val)
    # Create optimizer
    opt = optim.Adam(model.parameters(), lr=1e-4)
    # Log starting information
    log('Train started')
    log(f'Squeezenet with Adam optimizer, LR=1e-4, batch_size={batch_size}, epochs={num_epochs}')
    log(f'Train dataset: {TRAIN_DATASET_PATH}, val dataset: {VAL_DATASET_PATH}')
    # Loss per batch
    loss_mean = 0.0
    try:
        for epoch in range(num_epochs):
            if not adversarial_training:
                start_time = time.time()
                bg_iter = iter(train_bg)
                for step in range(int(len(train_dataset) / batch_size)):
                    model = model.train()
                    image_batch, emotion_batch = next(bg_iter)
                    image_batch = image_batch.float().cuda()
                    emotion_batch = emotion_batch.long().cuda()
                    opt.zero_grad()
                    out = model(image_batch)
                    loss = nn.CrossEntropyLoss()(out, torch.max(emotion_batch, 1)[1])
                    loss.backward()
                    opt.step()
                    loss_ = loss.cpu().data.numpy().item()
                    if step == 0:
                        pass
                    elif step % 30 == 0:
                        log(
                            f"Epoch: {epoch}/{num_epochs} Step: {step}/{int(len(train_dataset) / batch_size)} Loss: {loss_mean}")
                        loss_mean = 0.0
                    else:
                        loss_mean += loss_ / 30.0


            else:
                start_time = time.time()
                bg_iter = iter(train_bg)
                model = model.train()
                for step in range(int(len(train_dataset) / batch_size)):
                    model = model.train()
                    image_batch, emotion_batch = next(bg_iter)
                    image_batch = image_batch.float().cuda()
                    emotion_batch = emotion_batch.float().cuda()
                    opt.zero_grad()
                    delta = pgd_linf(model, image_batch, emotion_batch)
                    out = model(image_batch + delta)
                    loss = nn.CrossEntropyLoss()(out, torch.max(emotion_batch, 1)[1])
                    loss.backward()
                    opt.step()
                    loss_ = loss.cpu().data.numpy().item()
                    if step == 0:
                        pass
                    elif step % 30 == 0:
                        log(
                            f"Adversarial epoch: {epoch}/{num_epochs} Step: {step}/{int(len(train_dataset) / batch_size)} Loss: {loss_mean}")
                        loss_mean = 0.0
                    else:
                        loss_mean += loss_ / 30.0

            # Decrease learning rate each 10 epoch
            if (epoch + 1) % 10 == 0:
                for g in opt.param_groups:
                    g['lr'] = g['lr'] / 2.0
            # Model evaluation
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
                pred.extend(torch.argmax(out[:, :7], dim=1).tolist())
                true.extend(torch.argmax(emotion_batch, dim=1).tolist())
                acc += torch.sum(torch.argmax(out[:, :7], dim=1) == torch.argmax(emotion_batch, dim=1)).float() / float(
                    batch_size)

            loss_emotion_mean /= int(len(val_dataset) / batch_size)
            acc /= int(len(val_dataset) / batch_size)
            matrix = confusion_matrix(true, pred)

            log(f"Epoch: {epoch + 20} val loss : {loss_emotion_mean}; val acc: {acc}")
            log(f"Confusion matrix :\n {matrix}")
            log(f"Epoch time: {time.time()-start_time:.2f}")
            if adversarial_training:
                torch.save(model.state_dict(),
                           f"checkpoints/final/Adversarial_epoch_{epoch + 20}_loss_{loss_emotion_mean:.5f}_acc_{acc:.5f}.pth")
            else:
                torch.save(model.state_dict(),
                           f"checkpoints/final/Epoch_{epoch + 20}_loss_{loss_emotion_mean:.5f}_acc_{acc:.5f}.pth")

    except Exception as e:
        print(traceback.format_exc())
        log(f'Error: {e}', message_type='ERROR')


if __name__ == '__main__':

    # Create CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Create paths to train and val data environment variables
    TRAIN_DATASET_PATH = ['data/labels_fer_train_emotion.txt']
    VAL_DATASET_PATH = ['data/labels_fer_val_emotion.txt']

    LOG_FILE = 'logs/experiment_deep_emotion.log'

    # split([# 'data/labels_adience_gender.txt'])
    #       'data/labels_utk_gender.txt'])
        # 'data/labels_imdb_gender.txt',
        # 'data/labels_wiki_gender.txt'])
    batch_size = 64
    epochs = 40
    #model = squeezenet1_1(pretrained=False, num_classes=1000)
    #model.classifier[1] = nn.Conv2d(512, 7, kernel_size=1)
    #model.num_classes = 7
    #model.cuda()
    model = DeepEmotion(num_classes=7)
    model.cuda()
    train(model, batch_size, epochs, adversarial_training=False)
