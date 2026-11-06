import json
import torch
from torch.utils.data import Dataset, DataLoader


def collate_with_padding(features):
    return torch.nn.utils.rnn.pad_sequence(features)


""" Contains various functions to prepare data for different models"""

def load_data_binary(targ_json):
    with open(targ_json, 'r') as f:
         data = json.load(f)
    features = []
    labels = []
    for file_name, content in data.items():
        label = 1 if content['condition'] == 'intoxicated' else 0 # Have to be careful to keep using 'condiCtion'
        labels.append(label)
        sub_features = []
        for key, value in content.get('features', {}).items():
            sub_features.extend(value)
        features.append(sub_features)

    class_0 = 0
    class_1 = 0

    for i in labels:
        if i == 0:
            class_0 += 1
        elif i == 1:
            class_1 += 1
    #print(f"class_0: {class_0/len(labels)}")
    #print(f"class_1: {class_1/len(labels)}")
    return features, labels


class CustomDatasetALC(Dataset):
    """ load_data is a function to determine what kinds of features and labels we want to
    put it the model, in this case, we just need drunk and sober features.
    Add a function to prepare data for multiclass classification.
    """
    def __init__(self, load_data_func, targ_json):
        features_list, labels_list = load_data_func(targ_json)
        self.features = [torch.tensor(feature, dtype=torch.float32) for feature in features_list]
        self.labels = torch.tensor(labels_list, dtype=torch.float32)
        #self.labels = labels.clone().detach()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_dataloaders(
        train_json: str,
        test_json: str,
        load_data_func,
        batch_size: int,
        #collate_fn=collate_with_padding,
):
    """Creates training and testing(validation) DataLoaders.

    Args:
    train_json: Path to json file of training dataset.
    test_json: Path to json file of testing dataset.
    load_data_func: functions for returning different features and labels lists, used for custom Dataset.
    batch_size: Number of smaples per batch in each of the DataLoaders.
    collate_fn: padding different lengths. (Functions should be added later on in this file)
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader)
    """
    train_data = CustomDatasetALC(load_data_func, train_json)
    test_data = CustomDatasetALC(load_data_func, test_json)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        #collate_fn=collate_with_padding,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        #collate_fn=collate_with_padding,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader
