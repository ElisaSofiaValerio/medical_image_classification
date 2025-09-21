"""
In this file
We build a csv which contains the metadata related to our dataset and also the train test split.
"""
import shutil
from collections import defaultdict
from fileinput import filename
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image

import pandas as pd


# write the functions to
def read_image_labels(dir: Path):
    """
    create a dictionary as follows.
    Example: {'heart': [path_to_image, path_to_image, ...],
            'liver': [path_to_image, path_to_image, ...],
            'lung': [path_to_image, path_to_image, ...]}
    Read the Path of the png files as Path and store them in a list
    :param dir:
    :return: dictionary
    """
    d = defaultdict(list)
    for f in dir.rglob('*.png'):
        d[f.parent.name.split('_')[1]].append(f)

        # if "Heart" in str(f):
        #     d["Heart"].append(f)
        # elif "Liver" in str(f):
        #     d["Liver"].append(f)
        # else:
        #     d["Lung"].append(f)

    return d

def copy_files(file_list, split, cls):
    for file_path in file_list:
        dest_path = output_dir / split / cls / file_path.name
        shutil.copy2(file_path, dest_path)

def mlset_split_dataframe(image_dict: Dict):
    """
    Create an empty dataframe
    insert into the dataframe:
    - file name (from the dict)
    - Path (as string from the Dict)
    - label (key of the dictionary: Heart, Lung, Liver)
    - mlset (empty)

    Train test validation split. Use stratified split.
    Once you get the mlset of each file, put that info in the dataframe mlset column for the respective row
    :param image_dict:
    :return: dataframe
    """
    validation_rate = 0.2
    test_rate = 0.2
    cols = ['FileName', 'Path', 'Label', 'mlset']
    df = pd.DataFrame(columns=cols)

    for key in image_dict:
        for i in range(len(image_dict[key])):
            path = image_dict[key][i]
            name = path.name.split('.')[0]
            new_row = [{'FileName': name, 'Path': str(path), 'Label': key, 'mlset': 0}]
            new_row_df = pd.DataFrame(new_row)
            df = pd.concat([df, new_row_df], ignore_index = True)

        # df_key_splitting = df_key.copy()
        # validation_df_key = df_key_splitting.sample(frac = validation_rate)
        # valid_items = validation_df_key['FileName'].tolist()
        # for item in valid_items:
        #     i = df_key[((df_key.FileName == item))].index
        #     df_key_splitting.drop(i)
        #     df_key.loc[i, ['mlset']] = 'Validation'
        #     print()


    # now splitting the dataset into train, test and validation
    classes = ["Task02_Heart", "Task06_Lung", "Task03_Liver"]
    for split in ["train", "val", "test"]:
        for cls in classes:
            class_dir = input_dir / cls
            files = [str(x) for x in class_dir.glob("*")]  # All files in the organ folder

            # First split: train+val vs test
            trainval_files, test_files = train_test_split(files, test_size=test_rate, random_state=1)

            # Second split: train vs val (val deve essere 10% del totale, quindi relativo a trainval)
            val_relative_size = validation_rate / (1 - test_rate)
            train_files, val_files = train_test_split(
                trainval_files, test_size=val_relative_size, random_state=1
            )

            # for item in train_files:
            #     i = df[(df['Path'] == str(item))].index
            #     df.loc[i, ['mlset']] = 'Training'
            # for item in val_files:
            #     i = df[(df['Path'] == str(item))].index
            #     df.loc[i, ['mlset']] = 'Validation'
            # for item in test_files:
            #     i = df[(df['Path'] == str(item))].index
            #     df.loc[i, ['mlset']] = 'Test'

            df.loc[df['Path'].isin(train_files), 'mlset'] = 'training'
            df.loc[df['Path'].isin(val_files), 'mlset'] = 'validation'
            df.loc[df['Path'].isin(test_files), 'mlset'] = 'testing'


    # x = df[df['Label'] == 'Heart']
    # y = df[df['Label'] == 'Liver']
    # z = df[df['Label'] == 'Lung']

    # x = df[['FileName', 'Path']]
    # target = df[['Label']]
    # # x_train, x_test, target_train, target_test = train_test_split(x, target, test_size=0.2, random_state=1)
    # test_ratio = 0.2
    # n_splits = int(1 / test_ratio)
    # cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    # for i, (train_idxs, test_idxs) in enumerate(cv.split(x, target)):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_idxs}")
    #     print(f"  Test:  index={test_idxs}")


    return df

def preprocess(input_csv: Path, h5_path: Path):
    df = pd.read_csv(input_csv)
    df = df.drop('Unnamed: 0', axis=1)
    with h5py.File(h5_path, 'w') as f:# context manager
        dset = f.create_dataset("metadata", (df.shape[0], 4), h5py.string_dtype(encoding='utf-8'))
        dset[:] = df.to_numpy()

        # h5 file should be self-sufficient, not depend on directories. So, it should have both data and metadata
        image_dataset = f.create_dataset('images', (df.shape[0], 256, 256), np.float32)
        for i, row in df.iterrows():
            file_path = row['Path']
            img = Image.open(file_path)
            img = img.resize((256, 256)).convert('L') # from RGBA to greyscale
            image_dataset[i, ...] = np.array(img)




if __name__== "__main__":
    is_preprocessing = True
    if not is_preprocessing:
        input_dir = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data")
        # output_dir = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\splitted_data")
        image_label_dictionary = read_image_labels(input_dir)
        df = mlset_split_dataframe(image_label_dictionary)
        df.to_csv(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\metadata.csv")
    else:
        h5_path = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\preprocess.h5")
        input_csv = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\metadata.csv")
        preprocess(input_csv, h5_path)


