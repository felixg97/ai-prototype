import os
import sys
import time

import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils.utils import load_local_dataset_train_test, load_local_dataset_tf, preprocess_data_per_tfmodel
from utils.utils import create_premodel

from utils.constants import BASE_PATH
from utils.constants import SOURCE_DATASETS
from utils.constants import TF_MODELS

source_data_path = BASE_PATH + "data/source/"

# Run specific constants, other to find in utils/constants.py
### Overall stuff
RANDOM_STATE = 42
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (*TARGET_SIZE, 3)

### Pre-model stuff
BUILD_PREMODEL = True

### Classification layer stuff
BUILD_MODEL = True


def run_train_premodels_with_sourcedata():
    
    start = time.time()
    
    timestamp_string = time.gmtime(start)
    timestamp_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", timestamp_string)
    
    print("######################################")
    print("######################################")
    print(f"### Training of pre-models startet at: {timestamp_string} ###")
    
    save_path = BASE_PATH + "results/pre_models/" + timestamp_string + "/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("### Created directory: " + save_path + " ###")
    
    for dataset_name, orig_img_format, num_classes in SOURCE_DATASETS:
        
        print(f"### Switching to dataset: {dataset_name} ###")
        
        dataset_path = source_data_path + dataset_name + "/"
        
        train_ds = load_local_dataset_tf(dataset_path, target_size=TARGET_SIZE, 
                                subset="training", batch_size=32)
        test_ds = load_local_dataset_tf(dataset_path, target_size=TARGET_SIZE, 
                                subset="test", batch_size=32)
        
        train_ds = train_ds.take(5)
        test_ds = test_ds.take(5)
        
        for premodel in TF_MODELS:
            print(f"### Switching to pre-model: {premodel} ###")
            
            model_save_path = save_path + premodel + "/" + dataset_name + "/"
            
            model = create_premodel(
                premodel, 
                dataset_name,
                INPUT_SHAPE,
                num_classes,
                model_save_path,
                build=BUILD_PREMODEL
                )
            
            print(f"### Pre-model {premodel} instantiated. ###")
            
            train_preprocessed = preprocess_data_per_tfmodel(train_ds, model_name=premodel)
            test_preprocessed = preprocess_data_per_tfmodel(test_ds, model_name=premodel)
            model.fit_and_save_pre_model(train_preprocessed, test_preprocessed)
            
            print(f"### Pre-model {premodel} trained and saved ###")
    

def run_train_models_with_targetdata():
    start = time.time()

    timestamp_string = time.gmtime(start)
    timestamp_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", timestamp_string)



if __name__ == '__main__':
    run_train_premodels_with_sourcedata()