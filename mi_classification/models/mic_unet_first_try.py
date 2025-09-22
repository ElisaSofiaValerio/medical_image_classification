import os
from pathlib import Path
import pytorch_lightning as L

import h5py
import numpy as np
import pandas as pd
import torch
from monai.networks.nets import BasicUnet
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from mi_classification.models.custom_model import DisagioNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
label_encoding = {'Heart': 0, 'Lung': 1, 'Liver': 2}

class ClassificationDataset(Dataset):
    # Dataset is the parent class
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
        # we shuffle the indices so that for every epoch, the model will get the images in a different order.
        # in this way the model will not assume that the first image is always ('Heart') (ie)

        return self.indices


    def __getitem__(self, idx):
        # idx is not our indeces (coming from the dataframe with get_indices function). These idx are Pytorch context
        # idx is the index to access the element of the indices list (indices list is self.indices)
        image = self._h5['images'][self.indices[idx], ...]
        label = self.df.iloc[self.indices[idx], 2].decode("utf-8")
        # the model does not understand strings, so we should create a translating systems from string to number
        num_label = np.array(label_encoding[label])


        return image, num_label


class FancyModel(DisagioNet, L.LightningModule):
    # This class has 2 parents
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        x = batch[0].unsqueeze(1)
        out = self.forward(x)
        # we are going to use the softmax functon as activation fnction because it gives the probability of something occurring
        out_act = nn.Softmax()(out)
        loss = nn.CrossEntropyLoss()(out, batch[1].long()) # batch[1] is the list of labels
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0].unsqueeze(1)
        out = self.forward(x)
        # we are going to use the softmax functon as activation fnction because it gives the probability of something occurring
        out_act = nn.Softmax()(out)
        loss = nn.CrossEntropyLoss()(out, batch[1].long())  # batch[1] is the list of labels
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) # we use Adam as a backpropagation algh.
        return optimizer

def train_pytorch_model(h5_path):
    batch_size = 2
    train_dataset = ClassificationDataset(h5_path, 'training')
    val_dataset = ClassificationDataset(h5_path, 'validation')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    dis_model = FancyModel()
    print(FancyModel.__mro__)
    #compiled_model = torch.compile(dis_model)
    trainer = L.Trainer(max_epochs=2)
    trainer.fit(model=dis_model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

if __name__ == "__main__":
    h5_path = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\preprocess.h5")
    train_pytorch_model(h5_path)



