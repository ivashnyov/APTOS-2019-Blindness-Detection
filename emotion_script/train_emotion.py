import cv2
import numpy as np
import torch
import traceback
from datetime import datetime
from albumentations import CLAHE
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import time
import os
from tools.train_val_split import split_train_test
from tools.squeezenet import squeezenet1_1
from torchvision import transforms, models

from imgaug import augmenters as iaa
from sklearn.metrics import cohen_kappa_score

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(y_hat, y, weights='quadratic'))


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


scale = 300


def preprocessing(img, scale):
    x = img[int(img.shape[0] / 2), :, : ].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    image = cv2.resize(img, (0, 0), fx=s, fy=s)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aug = CLAHE(p=1)
    image = aug(image=image)['image']
    blurred = cv2.GaussianBlur(image, (0, 0), 10)
    image = cv2.addWeighted(image, 4, blurred, -4, 128)
    b = np.zeros(image.shape)
    cv2.circle(b, (int(image.shape[1] / 2), int(image.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    image = image * b + 128 * (1 - b)
    return cv2.resize(image, (244, 244))


# Convert dataset file into proper form for training
class DiabeticDataset(Dataset):
    def __init__(self, dataset_path, files, transform, shuffle=True):
        self.dataset = files
        self.transform = transform
        self.shuffle = shuffle
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image = cv2.imread(os.path.join(self.dataset_path, example[0] + '.png'))
        image = preprocessing(image, scale)

        if self.transform:
            image = self.transform(image)
        # cv2.imwrite(example[0] + '.png', image)
        image = image / 255.
        # image = np.expand_dims(image, axis=2)
        target = np.zeros(5)
        target[int(example[1])] = 1
        return image.transpose((2, 0, 1)), target


def calculate_weights(dataset):
    """
    Calculate weights of classes in dataset
    :param labels: list of files with parsed data
    :return: weights of classes
    """

    classes = {'NoDR': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0, 'Proliferative': 0}
    seq_type = []
    seq_weight = []
    for example in dataset:
        stage = example[1]
        if stage == '0':
            classes['NoDR'] += 1
            seq_type.append('NoDR')
        elif stage == '1':
            classes['Mild'] += 1
            seq_type.append('Mild')
        elif stage == '2':
            classes['Moderate'] += 1
            seq_type.append('Moderate')
        elif stage == '3':
            classes['Severe'] += 1
            seq_type.append('Severe')
        elif stage == '4':
            classes['Proliferative'] += 1
            seq_type.append('Proliferative')

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


def train(model, batch_size, num_epochs, train_data, val_data, adversarial_training=False):
    """
    Training model
    :param model: model to train
    :param batch_size: size of batch
    :param num_epochs: number of epochs
    :param adversarial_training: use or not adversarial training
    """

    # Create train dataset with augmentation
    train_dataset = DiabeticDataset(dataset_path='../../../../APTOS_2019_Blindness_Detection/train_images', files=train_data,
                                    transform=transforms.Compose([Flip()]))
    sampler_train = WeightedRandomSampler(weights=calculate_weights(train_data), num_samples=len(train_dataset))
    train_bg = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train)
    # Create validation dataset
    val_dataset = DiabeticDataset(dataset_path='../../../../APTOS_2019_Blindness_Detection/train_images', files=val_data, transform=False)
    sampler_val = WeightedRandomSampler(weights=calculate_weights(val_data), num_samples=len(val_dataset))
    val_bg = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val)
    # Create optimizer
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(opt, 3, gamma=0.4)
    # Log starting information
    log('Train started')
    log(f'Squeezenet with Adam optimizer, LR=1e-4, batch_size={batch_size}, epochs={num_epochs}')
    # Loss per batch
    loss_mean = 0.0
    try:
        for epoch in range(num_epochs):
            if not adversarial_training:
                scheduler.step()
                start_time = time.time()
                bg_iter = iter(train_bg)
                for step in range(int(len(train_dataset) / batch_size)):
                    model = model.train()
                    image_batch, emotion_batch = next(bg_iter)
                    image_batch = image_batch.float().cuda()
                    emotion_batch = emotion_batch.float().cuda()
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

            # Model evaluation
            model = model.eval()
            # Loss per batch on validation data
            loss_emotion_mean = 0.0
            bg_iter = iter(val_bg)
            acc = 0.0
            y_hat = []
            y = []
            for step in range(int(len(val_dataset) / batch_size)):
                image_batch, emotion_batch = next(bg_iter)
                image_batch = image_batch.float().cuda()
                emotion_batch = emotion_batch.float().cuda()
                out = model(image_batch)
                loss_emotion = nn.CrossEntropyLoss()(out, torch.max(emotion_batch, 1)[1])
                loss_emotion_mean += loss_emotion.cpu().data.numpy().item()
                y_hat.extend(torch.argmax(out[:, :7], dim=1).tolist())
                y.extend(torch.argmax(emotion_batch, dim=1).tolist())
                acc += torch.sum(torch.argmax(out[:, :7], dim=1) == torch.argmax(emotion_batch, dim=1)).float() / float(
                    batch_size)

            loss_emotion_mean /= int(len(val_dataset) / batch_size)
            acc /= int(len(val_dataset) / batch_size)
            kappa = quadratic_kappa(y_hat, y)

            log(f"Epoch: {epoch + 20} val loss : {loss_emotion_mean}; val acc: {acc}")
            log(f"Kappa :\n {kappa}")
            log(f"Epoch time: {time.time()-start_time:.2f}")
            if adversarial_training:
                torch.save(model.state_dict(),
                           f"checkpoints/Adversarial_epoch_{epoch}_loss_{loss_emotion_mean:.5f}_acc_{acc:.5f}.pth")
            else:
                torch.save(model.state_dict(),
                           f"checkpoints/Epoch_{epoch}_loss_{loss_emotion_mean:.5f}_acc_{acc:.5f}.pth")

    except Exception as e:
        print(traceback.format_exc())
        log(f'Error: {e}', message_type='ERROR')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    train_data, val_data = split_train_test(path_to_file='../../../../APTOS_2019_Blindness_Detection/train.csv',
                                            train_test_ratio=0.85,
                                            save=False)
    print(len(train_data), len(val_data))
    LOG_FILE = 'logs/second.log'

    batch_size = 32
    epochs = 150
    model = squeezenet1_1(pretrained=True, num_classes=1000)
    model.classifier[1] = nn.Conv2d(512, 5, kernel_size=1)
    model.num_classes = 5
    model.cuda()
    train(model, batch_size, epochs, train_data, val_data, adversarial_training=False)
