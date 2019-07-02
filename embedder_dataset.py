from torch.utils.data import Dataset
from PIL import Image
import os


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


class EmbedderDataset(Dataset):
    def __init__(self, dataset_path, files, transform, shuffle=True):
        self.dataset = files
        self.transform = transform
        self.shuffle = shuffle
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        image = Image.open(os.path.join(self.dataset_path, example[0] + '.png'))

        if self.transform:
            image = self.transform(image)

        sample = {'tensor': image, 'target': int(example[1])}

        return sample