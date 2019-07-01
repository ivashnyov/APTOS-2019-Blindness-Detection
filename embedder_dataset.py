from torch.utils.data import Dataset
from skimage import io

class EmbedderDataset(Dataset):
    def __init__(self, files, transform, shuffle):
        self.dataset = files
        self.transform = transform
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image = io.imread(example[0] + '.jpg')
        sample = {'tensor': image, 'target': int(example[1])}

        if self.transform:
            sample = self.transform(sample)

        return sample