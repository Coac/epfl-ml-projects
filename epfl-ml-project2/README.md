# Project Road Segmentation

## Required libraries
#### Keras
```
pip install keras
```

#### Tensorflow
Installation instructions:
https://www.tensorflow.org/install/install_windows

#### Scikit-learn
```
pip install -U scikit-learn
pip install scikit-image
```

## Datas
You need to include the dataset as follow :
```
datas
-- test_set_images
---- test_01
---- test_02
---- ...
---- test_50

-- training
---- groundtruth
------ satImage_001.png
------ satImage_002.png
------ ...
------ satImage_100.png

---- images
------ satImage_001.png
------ satImage_002.png
------ ...
------ satImage_100.png


```

## Run.py
This script runs the predictions on the test set images and create the submission file for Kaggle.
```
python run.py
```

## Notebook.ipynb
- CNN and Denoiser.ipynb
- Segnet Autoencoder.ipynb
- Sharr filtering.ipynb
- ZCA whitening.ipynb
- Fine tuning.ipynb
- Preprocessing.ipynb

## functions_v1.py
Contains all the functions used to run the code




