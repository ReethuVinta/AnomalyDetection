# CSE-CIC-IDS2018

## Dependencies 
- We suggest to create new virtual environment and install these dependencies. 
- ```pip install git+https://github.com/ContinualAI/avalanche.git```
- ```pip install pandas```
## Dataset
- Follow the instructions given in this [link](https://github.com/Kaggle/kaggle-api#api-credentials) for downloading the CSE-CIC-IDS2018 dataset from Kaggle API.
- Place the csv files in the folder named `data/`

## Model Architecture
- Default model architecure is `GEM`. `EWC` can also be used by passing it as a comand line argument. 

## To run the model 
- ```python preprocess_dataset.py && python [simpleMLP | CNN2D].py```

