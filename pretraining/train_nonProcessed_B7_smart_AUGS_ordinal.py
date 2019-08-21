import sys
import torch 
import os
import random
import torch.nn as nn
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
import sys
import collections
from torch.utils.data import Dataset, WeightedRandomSampler, SubsetRandomSampler, DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, CenterCrop, RandomRotate90, RandomCrop, 
    PadIfNeeded, Normalize, Flip, OneOf, Compose, Resize, Transpose, 
    IAAAdditiveGaussianNoise, GaussNoise, CLAHE, RandomBrightnessContrast, HueSaturationValue,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from catalyst.contrib.schedulers import OneCycleLR, ReduceLROnPlateau, StepLR, MultiStepLR
from catalyst.dl.experiment import SupervisedExperiment
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback, F1ScoreCallback, ConfusionMatrixCallback, MixupCallback
from catalyst.dl.core.state import RunnerState
from catalyst.dl.core import MetricCallback
from catalyst.dl.callbacks import CriterionCallback
from efficientnet_pytorch import EfficientNet
from  pretrainedmodels import resnext101_32x4d

def get_activation_fn(type='relu'):
    """
    Return tensorflow activation function given string name.
    Args:
        type:
    Returns:
    """
    if type == 'relu':
        return torch.relu
    elif type == 'elu':
        return torch.elu
    elif type == 'tanh':
        return torch.tanh
    elif type == 'Sigmoid':
        return torch.sigmoid
    elif type == 'softplus':
        return torch.softplus
    elif type == None:
        return None
    else:
        raise Exception("Activation function is not supported.")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
class DiabeticDataset(Dataset):
    def __init__(self, dataset_path, labels, ids, albumentations_tr, extens, shuffle=True):
        self.labels = labels
        self.ids = ids
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.albumentations_tr = albumentations_tr
        self.extens = extens
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        #imid = self.ids[index]
        image = cv2.imread(os.path.join(self.dataset_path, self.ids[index] + '.{}'.format(self.extens)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        if self.albumentations_tr:
            augmented = self.albumentations_tr(image=image)
            image = augmented['image']
        target = self.labels[index]
        return torch.from_numpy(image.transpose((2, 0, 1))).float(), torch.tensor(target).float()

def quadratic_weighted_kappa(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    outputs_sum = (outputs.cpu().detach().numpy()>=0.5).sum(axis=1)
    score = cohen_kappa_score(outputs_sum,torch.sum(targets,1).detach().cpu().numpy(),weights='quadratic')
    return score
class QuadraticKappScoreMetricCallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "qkappa_score",
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=quadratic_weighted_kappa,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )
def mean_squared_error_callback(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    outputs_sum = (outputs.cpu().detach().numpy()>=0.5).sum(axis=1)
    score = mean_squared_error(outputs_sum,torch.sum(targets,1).detach().cpu().numpy())
    return score
class MSECallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "mse_score",
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=mean_squared_error_callback,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )
        
def mean_absolute_error_callback(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    outputs_sum = (outputs.cpu().detach().numpy()>=0.5).sum(axis=1)
    score = mean_absolute_error(outputs_sum,torch.sum(targets,1).detach().cpu().numpy())
    return score
class MAECallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "mae_score",
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=mean_absolute_error_callback,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )
 
    
    
from typing import List
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

def aug_train(resolution,p=1): 
    return Compose([Resize(resolution,resolution),
                    OneOf([
                        HorizontalFlip(), 
                        VerticalFlip(), 
                        RandomRotate90(), 
                        Transpose()],p=0.5),
                    Normalize()
                    ], p=p)
def aug_train_heavy(resolution,p=1): 
    return Compose([Resize(resolution,resolution),
                    OneOf([
                        HorizontalFlip(), 
                        VerticalFlip(), 
                        RandomRotate90(), 
                        Transpose()],p=0.5),
                    OneOf([
                        IAAAdditiveGaussianNoise(),
                        GaussNoise(),
                    ], p=0.5),
                    OneOf([
                        MotionBlur(p=.2),
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ], p=0.5),
                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                    OneOf([
                        OpticalDistortion(p=0.3),
                        GridDistortion(p=.1),
                        IAAPiecewiseAffine(p=0.3),
                    ], p=0.5),
                    OneOf([
                        CLAHE(clip_limit=2),
                        IAASharpen(),
                        IAAEmboss(),
                        RandomBrightnessContrast(),
                    ], p=0.5),
                    HueSaturationValue(p=0.3),
                    Normalize()
                    ], p=p)
def aug_val(resolution,p=1): 
    return Compose([Resize(resolution,resolution),Normalize()], p=p)
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
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    root = "../../input/"
    train_path = root + 'old_train/train_test/'
    valid_path = root + 'train/'
    old_test = pd.read_csv(root + 'old_train/retinopathy_solution.csv')
    old_train = pd.read_csv(root + 'old_train/trainLabels.csv')
    train = pd.read_csv(root + 'train.csv')
    package_path = 'efficientnet'

    sys.path.append(package_path)
    num_classes = 4
    seed_everything(1234)
    lr          = 3e-5 # 3e-4
    IMG_SIZE    = 256
    BS          = 120   # 12
    runner = SupervisedRunner()
    model = resnext101_32x4d(num_classes=1000, pretrained='imagenet')
    dim_feats = model.last_linear.in_features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.cuda()

    total_old_data = pd.concat([old_train,old_test])
    train_ids = total_old_data['image'].values
    #train_labels = total_old_data['level'].values
    train_labels_oridnal = np.zeros((len(total_old_data['level']), 4))
    for idx, value in enumerate(total_old_data['level'].values):
        train_labels_oridnal[idx,:value]  = 1
    #new data
    val_ids = train['id_code'].values
    #val_labels = train['diagnosis'].values
    val_labels_oridnal = np.zeros((len(train['diagnosis']), 4))
    for idx, value in enumerate(train['diagnosis'].values):
        val_labels_oridnal[idx,:value]  = 1
    train_dataset = DiabeticDataset(dataset_path=train_path, 
                                    labels = train_labels_oridnal, 
                                    ids = train_ids, 
                                    albumentations_tr = aug_train_heavy(IMG_SIZE), 
                                    extens='jpeg') 
    val_dataset = DiabeticDataset(dataset_path=valid_path, 
                                  labels = val_labels_oridnal, 
                                  ids = val_ids, 
                                  albumentations_tr = aug_val(IMG_SIZE), 
                                  extens='png') 
    #class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    #weight = 1./class_sample_count
    #weight = dict(zip(np.unique(train_labels), weight))
    #samples_weight = np.array([weight[t] for t in train_labels])
    #sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader =  DataLoader(train_dataset,
                               num_workers=16,
                               pin_memory=False,
                               batch_size=BS,
                               shuffle=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=16,
                            pin_memory=False,
                            batch_size=BS)
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader
    logdir = 'logs/resnaxt101_regressionpretrain_ordinal_with_augs_256/'
    print('Training only head for 3 epochs with heavy augs and cutmix')
    for p in model.parameters():
        p.requires_grad = False
    for p in model.last_linear.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    num_epochs = 1
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            scheduler=scheduler,
            callbacks=[
                MixupCallback(),
                QuadraticKappScoreMetricCallback(),
                MSECallback(),
                MAECallback()
                      ],
            num_epochs=num_epochs,
            verbose=True
            )      
    print('Train whole net for 15 epochs with heavy augs and cutmix')
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    num_epochs = 9
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            scheduler=scheduler,
            callbacks=[
                MixupCallback(),
                QuadraticKappScoreMetricCallback(),
                MSECallback(),
                MAECallback(),               
                EarlyStoppingCallback(patience=7, metric='loss')
                      ],
            num_epochs=num_epochs,
            verbose=True
            ) 
    #Train with D4 and lower lr
    train_dataset = DiabeticDataset(dataset_path=train_path, 
                                    labels = train_labels_oridnal, 
                                    ids = train_ids, 
                                    albumentations_tr = aug_train(IMG_SIZE), 
                                    extens='jpeg') 
    val_dataset = DiabeticDataset(dataset_path=valid_path, 
                                  labels = val_labels_oridnal, 
                                  ids = val_ids, 
                                  albumentations_tr = aug_val(IMG_SIZE), 
                                  extens='png')    
    train_loader =  DataLoader(train_dataset,
                               num_workers=16,
                               pin_memory=False,
                               batch_size=BS,
                               shuffle=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=16,
                            pin_memory=False,
                            batch_size=BS)
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader
    print('Training only head for 3 epochs with D4 augs')
    for p in model.parameters():
        p.requires_grad = False
    for p in model.last_linear.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr/5, weight_decay=0.01)
    num_epochs = 1
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            callbacks=[
                QuadraticKappScoreMetricCallback(),
                MSECallback(),
                MAECallback(),               
                      ],
            num_epochs=num_epochs,
            verbose=True
            )      
    print('Train whole net for 15 epochs with D4 augs')
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr/5, weight_decay=0.01)
    num_epochs = 9
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=5)
    runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            scheduler=scheduler,
            callbacks=[
                QuadraticKappScoreMetricCallback(),
                MSECallback(),
                MAECallback(),               
                EarlyStoppingCallback(patience=7, metric='loss')
                      ],
            num_epochs=num_epochs,
            verbose=True
            )