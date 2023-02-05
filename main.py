import os
import sys
import time

import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils.utils import load_local_dataset_train_test, load_local_dataset_tf, preprocess_data_per_tfmodel
from utils.utils import create_premodel

from utils.constants import BASE_PATH
from utils.constants import SOURCE_DATASETS
from utils.constants import TARGET_DATASETS
from utils.constants import TF_MODELS

source_data_path = BASE_PATH + "data/source/"
target_data_path = BASE_PATH + "data/target/"

VERBOSE = True

# Run specific constants, other to find in utils/constants.py
### Overall stuff
RANDOM_STATE = 42
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (*TARGET_SIZE, 3)
BATCH_SIZE = 16

### Pre-model stuff
BUILD_PREMODEL = True

### Classification layer stuff
BUILD_MODEL = True
TARGET_ITERATIONS = 10
K_MAX = 100

############ Test stuff ############
TESTING = False
####################################


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
    
    for dataset_name, img_format, num_classes, orig_shape in SOURCE_DATASETS:
        
        print("######################################")
        print(f"### Switching to dataset: {dataset_name} ###")
        print("######################################")
        
        dataset_path = source_data_path + dataset_name + "/"
        
        print()
        print(dataset_path)
        print()
        
        print("######################################")
        train_ds = load_local_dataset_tf(dataset_path, target_size=TARGET_SIZE, 
                                subset="training", batch_size=BATCH_SIZE)
        
        print("######################################")
        test_ds = load_local_dataset_tf(dataset_path, target_size=TARGET_SIZE, 
                                subset="test", batch_size=BATCH_SIZE)
        print("######################################")
        
        if TESTING:
            train_ds = train_ds.take(5)
            test_ds = test_ds.take(5)
        
        for premodel in TF_MODELS:
            print("######################################")
            print(f"### Switching to pre-model: {premodel} ###")
            print("######################################")
            
            model_save_path = save_path + premodel + "_" + dataset_name + "/"
            
            model = create_premodel(
                model_save_path,
                premodel, 
                INPUT_SHAPE,
                dataset_name,
                num_classes,
                verbose = True
                )
            
            print(f"### Pre-model {premodel} instantiated. ###")
            print("######################################")
            
            train_preprocessed = preprocess_data_per_tfmodel(train_ds, model_name=premodel)
            test_preprocessed = preprocess_data_per_tfmodel(test_ds, model_name=premodel)

            try:
                model.fit(train_preprocessed, test_preprocessed)
            except Exception as e:
                print("######################################")            
                print("########## error occured #############")
                print(f"BATCH_SIZE: {BATCH_SIZE}")
                print(e)            
                print("######################################")            
            
            print("######################################")
            print(f"### Pre-model {premodel} trained and saved ###")
    

def run_train_models_with_targetdata():
    start = time.time()

    timestamp_string = time.gmtime(start)
    timestamp_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", timestamp_string)

    print("######################################")
    print("######################################")
    print(f"### Training of full models startet at: {timestamp_string} ###")
    
    
    # if not os.path.exists(save_path_models):
    #     os.makedirs(save_path_models)
    #     print("### Created directory: " + save_path_models + " ###")
        

    for dataset_name, img_format, num_classes, orig_shape in SOURCE_DATASETS:
        
        print("######################################")
        print(f"### Switching to dataset: {dataset_name} ###")
        print("######################################")
        
        for premodel in TF_MODELS:
            print("######################################")
            print(f"### Switching to pre-model: {premodel} ###")
            print("######################################")
            
            # create full model
            # model = create_full_model()
            
            print(f"### Pre-model {premodel} instantiated. ###")
            print("######################################")
            
            # create dataframe
        
            for iteration in range(TARGET_ITERATIONS):
                pass
            
                # load data set 
                # RANDOM_STATE + iteration 
            
                for k in range(K_MAX):
                    pass
                
                    # tain and test model

                    # return metrics, metrics_best
        


if __name__ == '__main__':
    run_train_premodels_with_sourcedata()