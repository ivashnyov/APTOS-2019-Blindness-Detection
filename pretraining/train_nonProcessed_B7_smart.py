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
        return torch.from_numpy(image.transpose((2, 0, 1))).float(), torch.tensor(np.expand_dims(target,0)).float()
def quadratic_weighted_kappa(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = None
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
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    outputs_clipped = list()
    outputs_clipped = np.rint(outputs)
    outputs_clipped[outputs_clipped<0] = 0
    outputs_clipped[outputs_clipped>4] = 4
    #for o in outputs:
    #    if o <= 0.5:
    #        outputs_clipped.append(0)
    #    if 0.5 > o <= 1.5:
    #        outputs_clipped.append(1)
    #    if 1.5 < o <= 2.5:
    #        outputs_clipped.append(2)
    #    if 2.5 < o <= 3.5:
    #        outputs_clipped.append(3)
    #    if o > 3.5:
    #        outputs_clipped.append(4)      
    #simple clip of outputs
    score = cohen_kappa_score(outputs_clipped, targets, weights='quadratic')
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
        activation: str = None
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
    activation: str = None
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

    outputs = outputs.cpu().detach().numpy()
    score = mean_squared_error(outputs, targets.detach().cpu().numpy())
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
        activation: str = None
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
    activation: str = None
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

    outputs = outputs.cpu().detach().numpy()
    score = mean_absolute_error(outputs, targets.detach().cpu().numpy())
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
        activation: str = None
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
def aug_train(resolution,p=1): 
    return Compose([Resize(resolution,resolution),
                    OneOf([
                        HorizontalFlip(), 
                        VerticalFlip(), 
                        RandomRotate90(), 
                        Transpose()],p=0.5),
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
    package_path = './kaggle/aptos/EfficientNet-PyTorch/efficientnet_pytorch'
    sys.path.append(package_path)
    num_classes = 1
    seed_everything(1234)
    lr          = 3e-4
    IMG_SIZE    = 256
    model = EfficientNet.from_pretrained('efficientnet-b7')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    model.cuda()
    old_test = pd.read_csv('/home/skolchen/kaggle/aptos/old_train/retinopathy_solution.csv')
    old_train = pd.read_csv('/home/skolchen/kaggle/aptos/old_train/trainLabels.csv')
    total_old_data = pd.concat([old_train,old_test])
    train_ids = total_old_data['image'].values
    train_labels = total_old_data['level'].values
    #new data
    train = pd.read_csv('/home/skolchen/kaggle/aptos/train.csv')
    val_ids = train['id_code'].values
    val_labels = train['diagnosis'].values
    train_dataset = DiabeticDataset(dataset_path='/home/skolchen/kaggle/aptos/old_train/train/train/', 
                                    labels = train_labels, 
                                    ids = train_ids, 
                                    albumentations_tr = aug_train(IMG_SIZE), 
                                    extens='jpeg') 
    val_dataset = DiabeticDataset(dataset_path='/home/skolchen/kaggle/aptos/train/', 
                                  labels = val_labels, 
                                  ids = val_ids, 
                                  albumentations_tr = aug_val(IMG_SIZE), 
                                  extens='png') 
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1./class_sample_count
    weight = dict(zip(np.unique(train_labels), weight))
    samples_weight = np.array([weight[t] for t in train_labels])
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader =  DataLoader(train_dataset,
                               num_workers=16,
                               pin_memory=False,
                               batch_size=12,
                               shuffle=True)
    val_loader = DataLoader(val_dataset,
                            num_workers=16,
                            pin_memory=False,
                            batch_size=12)
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader
    runner = SupervisedRunner()
    logdir = 'logs/efficient_net_b7_regressionpretrain_smart_v1/'
    print('Training only head for 5 epochs')
    for p in model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 5
    criterion = nn.MSELoss()
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
                EarlyStoppingCallback(patience=25, metric='loss')
                      ],
            num_epochs=num_epochs,
            verbose=True
            )      
    print('Train whole net for 10 epochs')
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 10
    criterion = nn.MSELoss()
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
                EarlyStoppingCallback(patience=25, metric='loss')
                      ],
            num_epochs=num_epochs,
            verbose=True
            ) 