# medical_image_classification

Preprocessing methods: 
- standard normalization
- z score normalization
- monai transforms (https://docs.monai.io/en/stable/transforms.html#dictionary-transforms)

Pytorch can process images in parallel (through the CalssificationDataset/DataLoader class).
We talk about BATCH of data.
(https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

Augmentation methods:
- Rotation
- Translation
- Scaling
- Contrast changes