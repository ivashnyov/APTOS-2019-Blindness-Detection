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
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from models import (
    SphereLoss, CosineLoss, ArcLoss, InsightLoss,
    FaceNet, FaceEmbedder
)
from embedder_dataset import EmbedderDataset
from load_data import split_train_test
from utils import get_logging_str, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/r.isachenko/facenet/matching/tmp/', type=str,
                    metavar='PATH', help='path to tmp folder with datasets')
parser.add_argument('-m', '--model', default='sphereface', type=str,
                    choices=['sphereface', 'arcface', 'cosineface', 'insightface'],
                    help='model to use')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='FLOAT', help='initial learning rate')
parser.add_argument("--train-test-ratio", default=0.9, type=float,
                    metavar='FRAC',
                    help="the fraction of image for training")
parser.add_argument('--classes', default=-1, type=int,
                    metavar='N', help='number of classes')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of tmp loading workers (default: 4)')
parser.add_argument('--embed-size', default=128, type=int, metavar='N',
                    help='emdedding dimensionality')
parser.add_argument("--devices", default="0", type=str, help='gpu devices to use')
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
exp_name = 'resnet50_{}'.format(date)

embedder = FaceEmbedder(models.resnet50(pretrained=True), args.embed_size)
model = FaceNet(embedder, num_classes=n_classes)

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

optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.0005)
milestones = [7, 12, 16, 20, 23, 25, 27, 29, 31, 33, 35, 37]
scheduler = MultiStepLR(optimizer, milestones, gamma=0.4)

if args.resume:
    scheduler.load_state_dict(checkpoint['scheduler'])
    scheduler.last_epoch = 21

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_dataset = EmbedderDataset(files=train_imgs,
                                transform=train_transform,
                                shuffle=True)
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=args.workers)

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_dataset = EmbedderDataset(files=test_imgs,
                               transform=test_transform,
                               shuffle=False)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         num_workers=args.workers)


def train(loader, model, criterion, optimizer, epoch, device):
    model.train()
    epoch_loss = 0.
    total, correct = 0., 0.

    for k, samples in enumerate(tqdm(loader, desc=f'epoch {epoch}', leave=False)):
        X, y = samples['tensor'].to(device), samples['target'].to(device)
        output = model(X)
        aloss = criterion(output, y)
        epoch_loss += aloss.item()
        predicted = torch.max(output.data, 1)[1]
        total += y.size(0)
        correct += float(predicted.eq(y.data).cpu().sum())
        optimizer.zero_grad()
        aloss.backward()

        optimizer.step()

    with open(os.path.join(os.path.dirname(__file__), 'logs', exp_name + '.log'), mode='a') as f:
        f.write('-' * 80 + '\n')
    return epoch_loss, 100. * correct / total


def test(loader, model, device):
    total, correct = 0., 0.
    model.eval()
    with torch.no_grad():
        for k, samples in enumerate(loader):
            X, y = samples['tensor'].to(device), samples['target'].to(device)

            output = model(X)
            predicted = torch.max(output.data, 1)[1]
            total += y.size(0)
            correct += float(predicted.eq(y.data).cpu().sum())

    return 100. * correct / total


if os.path.exists(os.path.join(os.path.dirname(__file__), 'runs', exp_name)):
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'runs', exp_name))

writer_train = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs', exp_name, 'accuracy_train'))
writer_test = SummaryWriter(os.path.join(os.path.dirname(__file__), 'runs', exp_name, 'accuracy_test'))

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
    if args.logging:
        os.system('telegram-send "{}"'.format(log_str))
    print(log_str)
    print('-' * 40)

    filesave = os.path.join(os.path.dirname(__file__), 'pt_checkpoints', f'model_{args.model}_{date}_epoch_{epoch}.pt')
    save_checkpoint(epoch, model.module, optimizer, scheduler, filesave)

writer_train.close()
writer_test.close()