import warnings

warnings.filterwarnings("ignore")
import torch
import os
import random
import torch.nn as nn
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold
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
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback, F1ScoreCallback, ConfusionMatrixCallback, \
    MixupCallback
from catalyst.dl.core.state import RunnerState
from catalyst.dl.core import MetricCallback
from catalyst.dl.callbacks import CriterionCallback
from efficientnet.model import EfficientNet


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
        # imid = self.ids[index]
        image = cv2.imread(os.path.join(self.dataset_path, self.ids[index] + '.{}'.format(self.extens)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        # blurred = cv2.GaussianBlur(image, (0, 0), 10)
        # image = cv2.addWeighted(image, 4, blurred, -4, 128)
        if self.albumentations_tr:
            augmented = self.albumentations_tr(image=image)
            image = augmented['image']
        target = self.labels[index]
        return torch.from_numpy(image.transpose((2, 0, 1))).float(), torch.tensor(np.expand_dims(target, 0)).float()


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
    outputs_clipped[outputs_clipped < 0] = 0
    outputs_clipped[outputs_clipped > 4] = 4
    # for o in outputs:
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
    # simple clip of outputs
    score = cohen_kappa_score(outputs_clipped, targets, weights='quadratic')
    if np.isnan(score):
        score = 1.
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


# make cutmix onlyfor same classes
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


class CutmixCallbackSameClasses(CriterionCallback):
    """
    Callback to do mixup augmentation.
    Paper: https://arxiv.org/abs/1710.09412
    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
            self,
            fields: List[str] = ("features",),
            alpha=1.0,
            on_train_only=True,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = list()
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        classes_in_batch = torch.unique(state.input[self.input_key]).cpu().numpy()
        batch_idx = np.arange(state.input[self.fields[0]].shape[0])
        # now make permutations per each class

        for idx, image_class in enumerate(classes_in_batch):
            # images with this class
            class_mask = (state.input[self.input_key].cpu().numpy() == image_class).squeeze()
            class_idx = batch_idx[class_mask]
            index = np.random.permutation(class_idx)
            index = torch.tensor(index, dtype=torch.long)
            class_idx = torch.tensor(class_idx, dtype=torch.long)
            index.to(state.device)
            class_idx.to(state.device)
            for f in self.fields:
                bbx1, bby1, bbx2, bby2 = rand_bbox(state.input[f].size(), self.lam)
                state.input[f][class_idx, :, bbx1:bbx2, bby1:bby2] = state.input[f][index, :, bbx1:bbx2, bby1:bby2]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y = state.input[self.input_key]

        loss = self.lam * criterion(pred, y)
        return loss


class MixupCallbackSameClass(CriterionCallback):
    """
    Callback to do mixup augmentation.
    Paper: https://arxiv.org/abs/1710.09412
    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
            self,
            fields: List[str] = ("features",),
            alpha=1.0,
            on_train_only=True,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        classes_in_batch = torch.unique(state.input[self.input_key]).cpu().numpy()
        batch_idx = np.arange(state.input[self.fields[0]].shape[0])
        # now make permutations per each class

        for idx, image_class in enumerate(classes_in_batch):
            # images with this class
            class_mask = (state.input[self.input_key].cpu().numpy() == image_class).squeeze()
            class_idx = batch_idx[class_mask]
            index = np.random.permutation(class_idx)
            index = torch.tensor(index, dtype=torch.long)
            class_idx = torch.tensor(class_idx, dtype=torch.long)
            index.to(state.device)
            class_idx.to(state.device)
            for f in self.fields:
                state.input[f][class_idx] = self.lam * state.input[f][class_idx] + (1 - self.lam) * state.input[f][
                    index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y = state.input[self.input_key]

        loss = self.lam * criterion(pred, y)
        return loss


def aug_train(resolution, p=1):
    return Compose([Resize(resolution, resolution),
                    OneOf([
                        HorizontalFlip(),
                        VerticalFlip(),
                        RandomRotate90(),
                        Transpose()], p=0.5),
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


def aug_val(resolution, p=1):
    return Compose([Resize(resolution, resolution), Normalize()], p=p)


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


if __name__ == '__main__':
    # package_path = './kaggle/aptos/EfficientNet-PyTorch/efficientnet_pytorch'
    # sys.path.append(package_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
    num_classes = 1
    seed_everything(1234)
    lr = 3e-5
    IMG_SIZE = 256

    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    duplicated_images = pd.read_csv('../data/duplicated_info.csv')
    duplicated_images['id_code'] = duplicated_images['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    # image_infos = pd.read_csv('/home/skolchen/kaggle/aptos/image_info.csv')
    # image_infos['id_code'] = image_infos['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    train = train.loc[[x not in duplicated_images['id_code'].values for x in train['id_code'].values], :]
    X = train['id_code'].values

    # X_train, X_val, y_train, y_val = train_test_split(X, train['diagnosis'].values, test_size=0.002, random_state=42,
    #                                                   stratify=train['diagnosis'].values)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = train['diagnosis'].values
    fold = 0
    for train_index, val_index in skf.split(X, y):
        fold += 1
        print(f"___________________Training {fold} fold_____________________")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print(f"Num train: {len(X_train)}; Num val: {len(X_val)}")
        model = EfficientNet.from_pretrained('efficientnet-b7')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)
        model.cuda()
        model.load_state_dict(torch.load(os.path.join('best_256.pth'))['model_state_dict'])

        train_dataset = DiabeticDataset(dataset_path='../data/train_images',
                                        labels=y_train,
                                        ids=X_train,
                                        albumentations_tr=aug_train(IMG_SIZE),
                                        extens='png')
        val_dataset = DiabeticDataset(dataset_path='../data/train_images',
                                      labels=y_val,
                                      ids=X_val,
                                      albumentations_tr=aug_val(IMG_SIZE),
                                      extens='png')

        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        weight = dict(zip(np.unique(y_train), weight))
        samples_weight = np.array([weight[t] for t in y_train])
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset,
                                  num_workers=16,
                                  pin_memory=False,
                                  batch_size=16,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                num_workers=16,
                                pin_memory=False,
                                batch_size=16)
        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = val_loader
        runner = SupervisedRunner()
        logdir = f"logs/efficient_net_b7_regression_pretrain_fold_{fold}"
        print('Training only head for 5 epochs')
        for p in model.parameters():
            p.requires_grad = False
        for p in model._fc.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
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
                MixupCallback(),
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
                MixupCallback(),
                EarlyStoppingCallback(patience=25, metric='loss')
            ],
            num_epochs=num_epochs,
            verbose=True
        )
        print('Training only head for 5 epochs without mixup')
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
        print('Train whole net for 20 epochs without mixup')
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_epochs = 20
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