# KDDCup 1999 dataset 

## Dependencies 
- We suggest to create new virtual environment and install these dependencies. 
- ```pip install avalanche && pip install pandas```
## Dataset
- Create a folder named `data/` in current working directory and cd into it
- Download datset by ``` wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz ```
- Unzip by ```gzip -d kddcup.data.gz > kddcup.data ```

## Model Architecture
- Default model architecure is `GEM`

## To run the model 
- ```python preprocess_dataset.py && python SimpleMLP.py```

