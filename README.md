# U-Net CNNs for biomedical image segmentation: a comparison of various models

This project aims at showing how well various U-Net models perform on a dataset containing ultrasound images of knees from real patients. Each patient has a swollen knee, meaning that there’s an excess of liquid collected around it, and my objective is to create a model which segmentsthis liquid. To do so, I used a particular type of convolutional neural network, known as U-Net, which has become very useful when dealing with biomedical images.

_Because of privacy reasons, unfortunately I am not able to show the ultrasound images of the patients. However, I provided the code and the metrics results of the various models used._  

The python files are the following:
- `preprocessing.py`: it includes all the file importing, creation of dataloaders and data augmentation.
- `utils.py`: it includes various variables and functions used in some other parts of the code.
- `models.py`: it includes all the architecture of the models.
- `training.py`: it includes the training loop.
- `evaluation.py`: it includes the evaluation loop, with the creation of the predicted masks.
