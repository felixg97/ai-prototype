import os
import sys
import time

import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils.utils import load_local_dataset_train_test
from utils.utils import create_premodel

from utils.constants import BASE_PATH
from utils.constants import SOURCE_DATASETS
from utils.constants import TF_MODELS

source_data_path = BASE_PATH + "data/source/"

# Run specific constants, other to find in utils/constants.py
### Overall stuff
RANDOM_STATE = 42
TARGET_SIZE = (350, 350)
INPUT_SHAPE = (350, 350, 3)

### Pre-model stuff
BUILD_PREMODEL = True

### Classification layer stuff
BUILD_MODEL = True


def run_train_premodels_with_sourcedata():
    
    start = time.time()
    
    timestamp_string = time.gmtime(start)
    timestamp_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", timestamp_string)
    
    print(f"### Training of pre-models startet at: {timestamp_string} ###")
    
    save_path = BASE_PATH + "results/pre_models/" + timestamp_string + "/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("### Created directory: " + save_path + " ###")
    
    for dataset_name, img_format, num_classes in SOURCE_DATASETS:
        
        print()
        print(f"### Switching to dataset: {dataset_name} ###")
        
        X_train, y_train, X_test, y_test = None, None, None, None
        
        if dataset_name != "imagenet":
            X_train, y_train, X_test, y_test = load_local_dataset_train_test(
                source_data_path + dataset_name,
                img_format,
                target_size=TARGET_SIZE,
                random_state=RANDOM_STATE,
            )
    
    
        for premodel in TF_MODELS:
            print(f"### Switching to pre-model: {premodel} ###")
            
            model = create_premodel(
                premodel, 
                dataset_name,
                INPUT_SHAPE,
                num_classes,
                save_path,
                build=BUILD_PREMODEL
                )
            
            print(f"### Pre-model {premodel} instantiated. ###")
            
            model.fit_and_save_pre_model(
                X_train, y_train, X_test=X_test, y_test=y_test)
            print(f"### Pre-model {premodel} trained and saved ###")
    




if __name__ == '__main__':
    run_train_premodels_with_sourcedata()