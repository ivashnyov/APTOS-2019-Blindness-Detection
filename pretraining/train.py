import cv2
import numpy as np
import torch
import traceback
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
import time
import os
from train_val_split import split_train_test_new, split_train_test_old
from efficientnet.model import EfficientNet
from torchvision import transforms, models
from efficientnet.utils import round_filters, efficientnet
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, Flip, OneOf, Compose, Resize, Transpose
)

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


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = crop_image_from_gray(image)
    # blurred = cv2.GaussianBlur(image, (0, 0), 10)
    # image = cv2.addWeighted(image, 4, blurred, -4, 128)
    return cv2.resize(image, (224, 224))


# Convert dataset file into proper form for training
class DiabeticDataset(Dataset):
    def __init__(self, dataset_path, files, transform, albumentations_tr, shuffle=True):
        self.dataset = files
        self.transform = transform
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.albumentations_tr = albumentations_tr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        if self.transform:
            image = cv2.imread(os.path.join(self.dataset_path, example[0] + '.jpeg'))
            if image is None:
                image = cv2.imread(os.path.join('../../../../old_aptos/test/', example[0] + '.jpeg'))
        else:
            image = cv2.imread(os.path.join(self.dataset_path, example[0] + '.png'))
        image = preprocessing(image)

        if self.albumentations_tr:
            augmented = self.albumentations_tr(image=image)
            image = augmented['image']
        # cv2.imwrite(example[0] + '.png', image)
        image = image / 255.
        # image = np.expand_dims(image, axis=2)
        target = int(example[1])
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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(model, batch_size, num_epochs, train_data, val_data, adversarial_training=False):
    """
    Training model
    :param model: model to train
    :param batch_size: size of batch
    :param num_epochs: number of epochs
    :param adversarial_training: use or not adversarial training
    """

    # Create train dataset with augmentation
    train_dataset = DiabeticDataset(dataset_path='../../../../old_aptos/train', files=train_data, transform=True,
                                    albumentations_tr=aug_train())

    sampler_train = WeightedRandomSampler(weights=calculate_weights(train_data), num_samples=len(train_dataset))
    train_bg = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train)
    # Create validation dataset
    val_dataset = DiabeticDataset(dataset_path='../../../../APTOS_2019_Blindness_Detection/train_images', files=val_data, transform=False, albumentations_tr=False)
    sampler_val = WeightedRandomSampler(weights=calculate_weights(val_data), num_samples=len(val_dataset))
    val_bg = DataLoader(val_dataset, batch_size=8, sampler=sampler_val)
    # Create optimizer
    opt = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(opt, 3, gamma=0.4)
    # Log starting information
    log('Train started')
    log(f'Squeezenet with Adam optimizer, LR=1e-4, batch_size={batch_size}, epochs={num_epochs}')
    # Loss per batch
    loss_mean = 0.0
    alpha = 1.0
    try:
        for epoch in range(num_epochs):
            if not adversarial_training:
                scheduler.step()
                start_time = time.time()
                bg_iter = iter(train_bg)
                for step in range(int(len(train_dataset) / batch_size)):
                    model = model.train()
                    image_batch, emotion_batch = next(bg_iter)
                    #cutmix quick and dirty
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(image_batch.shape[0])
                    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch.size(), lam)
                    image_batch[:, :, bbx1:bbx2, bby1:bby2] = image_batch[index, :, bbx1:bbx2, bby1:bby2]
                    #
                    image_batch = image_batch.float().cuda()
                    emotion_batch = emotion_batch.float().cuda().view(-1, 1)
                    opt.zero_grad()
                    out = model(image_batch)
                    out[out > 4.] = 4.
                    out[out < 0.] = 0.
                    #cutmix loss
                    y_a = emotion_batch
                    y_b = emotion_batch[index]
                    loss = lam * nn.MSELoss()(out, y_a) + (1 - lam) * nn.MSELoss()(out, y_b)
                    #
                    #loss = nn.MSELoss()(out, emotion_batch)
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
                    loss = nn.CrossEntropyLoss()(out[0], torch.max(emotion_batch, 1)[1])
                    loss.backward()
                    opt.step()
                    loss_ = loss.cpu().data.numpy().item()
                    if step == 0:
                        pass
                    elif step % 2 == 0:
                        log(
                            f"Adversarial epoch: {epoch}/{num_epochs} Step: {step}/{int(len(train_dataset) / batch_size)} Loss: {loss_mean}")
                        loss_mean = 0.0
                    else:
                        loss_mean += loss_ / 2.0
            
            torch.save(model.state_dict(),
                           f"checkpoints/check_{epoch}.pth")
            # Model evaluation
            model = model.eval()
            # Loss per batch on validation data
            loss_emotion_mean = 0.0
            bg_iter = iter(val_bg)
            acc = 0.0
            y_hat = []
            y = []
            for step in range(int(len(train_dataset) / 8)):
                image_batch, emotion_batch = next(bg_iter)
                image_batch = image_batch.float().cuda()
                emotion_batch = emotion_batch.float().cuda().view(-1, 1)
                out = model(image_batch)
                out[out > 3.5] = 4.
                out[out <= 0.5] = 0.
                for o in out:
                    if 0.5 < o[0] <= 1.5:
                        o[0] = 1.
                    if 1.5 < o[0] <= 2.5:
                        o[0] = 2.
                    if 2.5 < o[0] <= 3.5:
                        o[0] = 3.
                loss_emotion = nn.MSELoss()(out, emotion_batch)
                loss_emotion_mean += loss_emotion.cpu().data.numpy().item()
                y_hat.extend(out.tolist())
                y.extend(emotion_batch.tolist())
                acc += torch.sum(out == emotion_batch).float() / float(
                    batch_size)

            loss_emotion_mean /= int(len(val_dataset) / 8)
            acc /= int(len(val_dataset) / 8)
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

        
def aug_train(p=1): 
    return Compose([OneOf([
                        HorizontalFlip(), 
                        VerticalFlip(), 
                        RandomRotate90(), 
                        Transpose()],p=0.25)
                    ], p=p)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

    train_data = split_train_test_old(path_to_file_test='../../../../old_aptos/retinopathy_solution.csv', path_to_file_train='../../../../old_aptos/trainLabels.csv')
    val_data, _ = split_train_test_new(path_to_file='../../../../APTOS_2019_Blindness_Detection/train.csv',
                                            train_test_ratio=1,
                                            save=False)
    print(len(train_data), len(val_data))
    LOG_FILE = 'logs/efficient_net.log'

    batch_size = 64
    epochs = 20
    model = EfficientNet.from_pretrained('efficientnet-b0')
    w, d, s, p = 1.0, 1.0, 224, 0.2
    blocks_args, global_params = efficientnet(
        width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    out_channels = round_filters(1280, global_params)
    model._fc = nn.Linear(out_channels, 1)
    model = nn.DataParallel(model)
    model.cuda()
    train(model, batch_size, epochs, train_data, val_data, adversarial_training=False)
