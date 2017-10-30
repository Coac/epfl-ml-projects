# EPFL Machine learning Project 1

The aim of this project is to develop and test a machine learning model meant to solve the challenge of finding the Higgs Boson. 
The dataset contains several features that describe collision events. From that data we predict whether each event originated a Higgs Boson.
This project takes the form of a [Kaggle competition](https://www.kaggle.com/c/epfml-higgs/) based on the Higgs Boson Machine Learning Challenge (2014).

You can have a look to our Jupyter notebook [project.ipynb](https://github.com/Coac/epfl-ml-projects/blob/master/epfl-ml-project1/project.ipynb) to see how we achieve the classification task.

There is also a report describing precisely our methods.

## Usage
To generate the submission file in the ``datas/`` folder
```
python run.py
```

## Project structure

### cross_validation.py
Cross validation with ridge regression

### feature_selection.py
Calculate mutual information and reorder the features

### features_engineering.py
Build complex models via polynomial, functions and combinations

### group_by.py
Group by specific categorical columns and by NaN columns

### helpers.py
Various methods to load csv, generate submission file, predict labels, get accuracy

### implementation.py
Contains all the machine learning methods : 
- least_squares_GD
- least_squares_SGD
- least_squares
- ridge_regression
- logistic_regression
- reg_logistic_regression


### pre_processing.py
Normalize, remove NaN columns

### knn.py
k-nearest neighbors, not used but working, it's just too slow
