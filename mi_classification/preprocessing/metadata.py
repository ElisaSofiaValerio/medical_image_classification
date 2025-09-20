"""
In this file
We build a csv which contains the metadata related to our dataset and also the train test split.
"""
from collections import defaultdict
from fileinput import filename
from pathlib import Path
from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from monai.handlers.mlflow_handler import pandas


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
        if "Heart" in str(f):
            d["Heart"].append(f)
        elif "Liver" in str(f):
            d["Liver"].append(f)
        else:
            d["Lung"].append(f)

    return d



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
    cols = ['FileName', 'Path', 'Label', 'mlset']
    df = pd.DataFrame(columns=cols)

    for key in image_dict:
        for i in range(len(image_dict[key])):
            path = str(image_dict[key][i])
            label = key
            name = path.split(key)[1]
            name = name[1:-4]
            new_row = [{'FileName': name, 'Path': path, 'Label': label, 'mlset': 0}]
            new_row_df = pd.DataFrame(new_row)
            df = pd.concat([df, new_row_df], ignore_index = True)

    # now splitting the dataset


    return df




if __name__== "__main__":
    input_dir = Path(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data")
    image_label_dictionary = read_image_labels(input_dir)
    df = mlset_split_dataframe(image_label_dictionary)
    df.to_csv(r"C:\Users\elisa\GitHub_projects\medical_image_classification\data\metadata.csv")

