import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
label_encoding = {'Heart': 0, 'Lung': 1, 'Liver': 2}

class ClassificationDataset(Dataset):
    def __init__(self, h5_file, mlset):
        self.h5_file = h5_file
        self._h5 = h5py.File(self.h5_file, 'r')
        self.mlset = mlset
        arr = self._h5['metadata'][:]
        self.df = pd.DataFrame(arr)
        self.get_indices()

    def __len__(self):
        return self.indices.shape[0]

    def get_indices(self):
        self.indices = self.df.loc[self.df[3] == self.mlset.encode(encoding="utf-8")].index.to_numpy()
        np.random.shuffle(self.indices)

        return self.indices


    def __getitem__(self, idx):
        image = self._h5['images'][self.indices[idx], ...]
        label = self.df.iloc[self.indices[idx], 2].decode("utf-8")
        # the model does not understand strings, so we should create a translating systems from string to number
        num_label = np.array(label_encoding[label])


        return image, num_label




if __name__ == "__main__":
    h5_path = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\preprocess.h5")
    train_dataset = ClassificationDataset(h5_path, 'training')
    train_dataset.__getitem__(2)

