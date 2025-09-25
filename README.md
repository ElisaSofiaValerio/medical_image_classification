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

Next steps:
- write a testing pipeline
- learn to create a network from an input shape 
to reach a target shape using as less parameters as possible
(not linear layers, or just small linear layers)