import os
from os import mkdir
from pathlib import Path
import pytorch_lightning as L

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import loggers as pl_loggers

from mi_classification.models.custom_model import DisagioNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
label_encoding = {'Heart': 0, 'Lung': 1, 'Liver': 2}
label_decoding = {value: key for key, value in label_encoding.items()}

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
        # we are going to use the softmax functon as activation function because it gives the probability of something occurring
        # we need the activation function to make some sense out of the logits (raw data) returned by the model
        out_act = nn.Softmax()(out)
        out_max = torch.argmax(out_act, dim=1).detach().numpy()
        labels = [label_decoding[i] for i in out_max]
        loss = nn.CrossEntropyLoss()(out, batch[1].long()) # batch[1] is the list of labels
        self.log('train_loss', loss)
        self.write_image(batch_idx, x, labels, 0)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0].unsqueeze(1)
        out = self.forward(x)
        out_act = nn.Softmax()(out)
        out_max = torch.argmax(out_act, dim=1).detach().numpy()
        labels = [label_decoding[i] for i in out_max]
        loss = nn.CrossEntropyLoss()(out, batch[1].long())  # batch[1] is the list of labels
        self.log('val_loss', loss)
        self.write_image(batch_idx, x, labels, 1)
        return loss

    def write_image(self, batch_idx, images, predicted_label, mlset_idx):
        # we do not write images every step because it takes too much disk space. We write every 10 steps (ex)
        k = 10 if mlset_idx == 0 else 1
        if batch_idx%k == 0:
            np_images = images.detach().numpy()
            first_image = np_images[0][0]
            first_label = predicted_label[0]
            im = Image.fromarray(first_image) # we need to use pillow to get the image as an actual image and not a numpy array
            ImageDraw.Draw(im).text(
                (20, 20), # location over the image where the text is going to be displayed
                first_label, # tect being typed
                255 # color. It is going to be White
            )
            final_image = np.asarray(im).reshape(1,256,256)
            normalized_final_image = (final_image - np.min(final_image)) / (np.max(final_image) - np.min(final_image))
            self.loggers[mlset_idx].experiment.add_image(f"image_{batch_idx}", normalized_final_image, self.current_epoch)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) # we use Adam as a backpropagation algh.
        return optimizer

def train_pytorch_model(h5_path, model_name: str):
    batch_size = 2
    train_dataset = ClassificationDataset(h5_path, 'training')
    val_dataset = ClassificationDataset(h5_path, 'validation')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    tb_logger = [pl_loggers.TensorBoardLogger(save_dir=fr"C:\Users\elisa\GitHub_projects\medical_image_classification\{model_name}\training"),
                 pl_loggers.TensorBoardLogger(save_dir=fr"C:\Users\elisa\GitHub_projects\medical_image_classification\{model_name}\validation")]
    dis_model = FancyModel()
    print(FancyModel.__mro__)
    #compiled_model = torch.compile(dis_model)
    trainer = L.Trainer(max_epochs=20, logger=tb_logger, reload_dataloaders_every_n_epochs=1)
    trainer.fit(model=dis_model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

def first_test_pytorch_model(h5_path, model_path: str):
    parent_path = h5_path.parent / "test_output"
    parent_path.mkdir(exist_ok=True)
    for L in label_encoding:
        (parent_path / L).mkdir(exist_ok=True)
    batch_size = 1
    test_dataset = ClassificationDataset(h5_path, 'testing')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    # we are now giving the weights we got from the training as ready-in inputs for the model
    # Lightning way:
    dis_model = FancyModel.load_from_checkpoint(model_path)
    # Pytorch way:
    # dis_model = FancyModel()
    # weights = torch.load(model_path)
    # dis_model.load_state_dict(weights['state_dict'])

    c = 0
    # we want to evaluate the model itself, discharging information about the connections between layers
    dis_model.eval() # setting the model to evaluation mode
    preds = []
    targets = []
    for x, label in test_data_loader:
        # the for loop is calling the get_item function of the ClassificationDataset class
        x = x.unsqueeze(1)
        out = dis_model(x) # this is equal to calling the forward function of dis_model. It is equal to: dis_model.forward(x)
        out_act = nn.Softmax()(out)
        out_max = torch.argmax(out_act, dim=1).detach().numpy()
        predicted_label = [label_decoding[i] for i in out_max]
        img = x.detach().numpy().reshape((256, 256)) # reshaping the image to just 256,256 without channel and batch info
        img = Image.fromarray(img)
        img = img.convert("L") # converting into a greyscale image because PIL only deals with those
        img.save(parent_path / predicted_label[0] / f"image_{c}.png")
        c += 1
        actual_label = label.detach().numpy()
        actual_label = [label_decoding[i] for i in actual_label]
        preds.append(predicted_label[0])
        targets.append(actual_label[0])
    print(classification_report(targets, preds))



if __name__ == "__main__":
    model_name = 'model_1'
    h5_path = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\preprocess.h5")
    train_pytorch_model(h5_path, model_name)
    model_path = r"C:\Users\elisa\GitHub_projects\medical_image_classification\model_1\training\lightning_logs\version_0\checkpoints\epoch=19-step=1280.ckpt"
    first_test_pytorch_model(h5_path, model_path)




