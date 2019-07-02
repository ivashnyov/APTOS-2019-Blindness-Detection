import numpy as np
import os
import datetime
import shutil
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import WeightedRandomSampler
from tensorboardX import SummaryWriter

from models import (
    SphereLoss, CosineLoss, ArcLoss, InsightLoss,
    FaceNet, FaceEmbedder
)

from squeezenet import squeezenet1_1

from embedder_dataset import EmbedderDataset, calculate_weights
from load_data import split_train_test
from utils import quadratic_kappa

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', default='/mnt/dataserver/inbox/APTOS 2019 Blindness Detection/train_images', type=str,
                    metavar='IMG_PATH', help='path to images folder')
parser.add_argument('--data_path', default='/mnt/dataserver/inbox/APTOS 2019 Blindness Detection/train.csv', type=str,
                    metavar='DATA_PATH', help='path to tmp csv file')
parser.add_argument('-m', '--model', default='sphereface', type=str,
                    choices=['sphereface', 'arcface', 'cosineface', 'insightface'],
                    help='model to use')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='FLOAT', help='initial learning rate')
parser.add_argument("--train-test-ratio", default=0.85, type=float,
                    metavar='FRAC',
                    help="the fraction of image for training")
parser.add_argument('--classes', default=5, type=int,
                    metavar='N', help='number of classes')
parser.add_argument('--embed-size', default=128, type=int, metavar='N',
                    help='emdedding dimensionality')
parser.add_argument("--devices", default="4, 5, 6", type=str, help='gpu devices to use')
parser.add_argument("--logging", action="store_true",
                    help="whether to log results to tg")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()

gpu_devices = list(map(int, args.devices.split(',')))

np.random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_devices))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = args.classes

train_imgs, test_imgs = split_train_test(path_to_file=args.data_path,
                                         train_test_ratio=args.train_test_ratio,
                                         save=True)
print('n_classes:', n_classes)
print('train_size:', len(train_imgs))
print('test_size:', len(test_imgs))
print('-' * 40)

date = datetime.date.today()
exp_name = 'squeezenet_{}'.format(date)

#embedder = FaceEmbedder(models.resnet50(pretrained=True), args.embed_size)
#model = FaceNet(embedder, num_classes=n_classes)

model = squeezenet1_1(pretrained=False)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        ckpt_n_classes = checkpoint['state']['fc_angle.weight'].shape[1]
        if n_classes == ckpt_n_classes:
            state = checkpoint['state']
            # остатки прошлой жизни
            state['fc_theta.weight'] = state.pop('fc_angle.weight')
            model.load_state_dict(state)
            print("=> sphereface model loaded (epoch {})".format(checkpoint['epoch']))
        else:
            embedder_state = {k[9:]: v for (k, v) in checkpoint['state'].items() if k.startswith('embedder.')}
            model.embedder.load_state_dict(embedder_state)
            print("=> sphereface embedder loaded (epoch {})".format(checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

if args.model == 'sphereface':
    criterion = SphereLoss(m=4)
elif args.model == 'arcface':
    criterion = ArcLoss(s=64., m=0.4)
elif args.model == 'cosineface':
    criterion = CosineLoss(s=64., m=0.2)
elif args.model == 'insightface':
    criterion = InsightLoss(s=64., m1=1., m2=0.2, m3=0.15)

model = model.to(device)
criterion = criterion.to(device)
model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), args.lr)
scheduler = StepLR(optimizer, 3, gamma=0.4)

if args.resume:
    scheduler.load_state_dict(checkpoint['scheduler'])
    scheduler.last_epoch = 21

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(1000, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_sampler = WeightedRandomSampler(weights=calculate_weights(train_imgs), num_samples=len(train_imgs))
train_dataset = EmbedderDataset(dataset_path=args.images_path,
                                files=train_imgs,
                                transform=train_transform)
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          sampler=train_sampler)


test_transform = transforms.Compose([
    transforms.Resize([1000, 1000]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_sampler = WeightedRandomSampler(weights=calculate_weights(test_imgs), num_samples=len(test_imgs))
test_dataset = EmbedderDataset(dataset_path=args.images_path,
                               files=test_imgs,
                               transform=test_transform)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         sampler=test_sampler)


def train(loader, model, criterion, optimizer, epoch, device):
    model.train()
    epoch_loss = 0.
    y1 = []
    y2 = []

    for k, samples in enumerate(tqdm(loader, desc=f'epoch {epoch}', leave=False)):
        X, y = samples['tensor'].to(device), samples['target'].to(device)
        output = model(X)
        # aloss = criterion(output, y)
        aloss = nn.CrossEntropyLoss()(output, y)
        epoch_loss += aloss.item()
        y1.extend(torch.max(output.data, 1)[1].tolist())
        y2.extend(y.data.tolist())
        optimizer.zero_grad()
        aloss.backward()
        optimizer.step()

    kappa = quadratic_kappa(y1, y2)
    with open(os.path.join(os.path.dirname(__file__), 'logs', exp_name + '.log'), mode='a') as f:
        f.write('-' * 80 + '\n')
    return epoch_loss, kappa


def test(loader, model, device):
    y1 = []
    y2 = []
    model.eval()
    with torch.no_grad():
        for k, samples in enumerate(loader):
            X, y = samples['tensor'].to(device), samples['target'].to(device)

            output = model(X)
            y1.extend(torch.max(output.data, 1)[1].tolist())
            y2.extend(y.data.tolist())

    kappa = quadratic_kappa(y1, y2)

    return kappa


if os.path.exists(os.path.join(os.path.dirname(__file__), 'runs', exp_name)):
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'runs', exp_name))

writer_train = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs', exp_name, 'kappa_train'))
writer_test = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs', exp_name, 'kappa_test'))

for epoch in range(args.start_epoch, args.epochs):
    scheduler.step()
    loss, train_acc = train(train_loader, model, criterion,
                            optimizer, epoch, device)
    loss = loss * args.batch_size / len(train_imgs)
    test_acc = test(test_loader, model, device)

    writer_train.add_scalar('tmp/accuracy', train_acc, epoch)
    writer_test.add_scalar('tmp/accuracy', test_acc, epoch)
    writer_train.add_scalar('tmp/loss_total', loss, epoch)

    log_str = '\n'.join([args.model,
                         'epoch: {}'.format(epoch),
                         'lr: {:.5f}'.format(scheduler.get_lr()[0]),
                         'train accuracy: {:.3f} %'.format(train_acc),
                         'test accuracy: {:.3f} %'.format(test_acc),
                         'loss: {:.3f}'.format(loss),
                         ])
    with open(os.path.join(os.path.dirname(__file__), 'logs', exp_name + '.log'), mode='a') as f:
        f.write(log_str)
    if args.logging:
        os.system('telegram-send "{}"'.format(log_str))
    print(log_str)
    print('-' * 40)

    filesave = os.path.join(os.path.dirname(__file__), 'pt_checkpoints', f'model_{args.model}_{date}_epoch_{epoch}.pt')
    torch.save(model.state_dict(), filesave)

writer_train.close()
writer_test.close()