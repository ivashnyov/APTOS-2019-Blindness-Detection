import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from embeded_script.embedder_dataset import EmbedderDataset
from embeded_script.load_data import split_train_test
import os
from tqdm import tqdm

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 1000, 1000)
    return x


train_imgs, test_imgs = split_train_test(path_to_file='/mnt/dataserver/inbox/APTOS 2019 Blindness Detection/train.csv',
                                         train_test_ratio=1,
                                         save=True)

num_epochs = 100
batch_size = 8
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.Resize([1000, 1000]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = EmbedderDataset(dataset_path='/mnt/dataserver/inbox/APTOS 2019 Blindness Detection/train_images',
                          files=train_imgs,
                          transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Create CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

model = autoencoder()
if torch.cuda.is_available():
    model.cuda()

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for k, data in enumerate(tqdm(dataloader, desc=f'epoch {epoch}', leave=False)):
        img = data
        img = Variable(img['tensor']).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

        torch.save(model.state_dict(), './conv_autoencoder.pth')