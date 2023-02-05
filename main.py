import os
import sys
import time

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from utils.utils import load_local_dataset_train_test, load_local_dataset_tf, preprocess_data_per_tfmodel
from utils.utils import create_premodel, create_full_model

from utils.constants import BASE_PATH
from utils.constants import SOURCE_DATASETS, TARGET_DATASETS
from utils.constants import TF_MODELS

source_data_path = BASE_PATH + "data/source/"
TARGET_DATA_PATH = BASE_PATH + "data/target/"

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
TESTING = True
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
        train_ds = None
        if dataset_name != "imagenet":
            train_ds = load_local_dataset_tf(dataset_path, target_size=TARGET_SIZE, 
                                subset="training", batch_size=BATCH_SIZE)
        
        print("######################################")
        test_ds = None
        if dataset_name != "imagenet":
            test_ds = load_local_dataset_tf(dataset_path, target_size=TARGET_SIZE, 
                                    subset="test", batch_size=BATCH_SIZE)
        print("######################################")
        
        if TESTING and dataset_name != "imagenet":
            train_ds = train_ds.take(5)
            test_ds = test_ds.take(5)
        
        for premodel in TF_MODELS:
            print("######################################")
            print(f"### Switching to pre-model: {premodel} ###")
            print("######################################")
            
            model = create_premodel(
                save_path,
                premodel, 
                INPUT_SHAPE,
                dataset_name,
                num_classes,
                verbose = False
                )
            
            print(f"### Pre-model {premodel} instantiated. ###")
            print("######################################")
            
            train_preprocessed = None
            test_preprocessed = None
            
            if dataset_name != "imagenet":
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
    
    
    weights_path = BASE_PATH + "results/pre_models/weights/"
    target_data_path = TARGET_DATA_PATH + "mechanicalseals_fulllight/"
    experiments_path = BASE_PATH + "results/experiments/"
    
    # if not os.path.exists(save_path_models):
    #     os.makedirs(save_path_models)
    #     print("### Created directory: " + save_path_models + " ###")
        

    for source_dataset_name, source_img_format, source_num_classes, source_orig_shape in SOURCE_DATASETS:
        
        print("######################################")
        print(f"### Switching to source dataset: {source_dataset_name} ###")
        print("######################################")
        
        for premodel in TF_MODELS:
            print("######################################")
            print(f"### Switching to pre-model: {premodel} ###")
            print("######################################")
            
            # print(f"### Pre-model {premodel} instantiated. ###")
            # print("######################################")
            
            for target_dataset_name, target_img_format, target_num_classes, target_orig_shape in TARGET_DATASETS:
                
                print("######################################")
                print(f"### Switching to target dataset: {target_dataset_name} ###")
                print("######################################")
                
                # path to pre trained models
                model_source_weights_path = weights_path + premodel + "_" + source_dataset_name + "/"
                
                for iteration in range(TARGET_ITERATIONS):
                    
                    # assembled name of experiment results per iteration per model per datasets
                    iter_experiments_name = experiments_path + "it_" + str(iteration) + \
                        premodel + "_" + source_dataset_name + "_" + target_dataset_name
                    
                    
                    # create list for later dataframe
                    experimental_results = []
                    
                    # load data set 
                    # w/ seed RANDOM_STATE + iteration 
                    train_ds = load_local_dataset_tf(
                        target_data_path, 
                        target_size=TARGET_SIZE, 
                        subset="training", 
                        batch_size=BATCH_SIZE,
                        seed=RANDOM_STATE+iteration
                    )
                    
                    test_ds = load_local_dataset_tf(
                        target_data_path, 
                        target_size=TARGET_SIZE, 
                        subset="test", 
                        batch_size=BATCH_SIZE,
                        seed=RANDOM_STATE+iteration
                    )
                    
                    train_preprocessed = preprocess_data_per_tfmodel(train_ds, model_name=premodel)
                    test_preprocessed = preprocess_data_per_tfmodel(test_ds, model_name=premodel)
                    
                
                    for k_shot in range(K_MAX):
                        
                        # create model save path
                        k_shot_model_save_path = experiments_path + "/models/" + \
                            "it_" + str(iteration) + premodel + "_" + source_dataset_name + \
                            "_" + target_dataset_name + "_kshot_" + str(k_shot) + "/"
                        
                        # create full model
                        model = create_full_model(
                            model_source_weights_path,
                            premodel,
                            INPUT_SHAPE,
                            source_dataset_name,
                            target_dataset_name,
                            target_num_classes,
                            k_shot,
                            verbose=True
                        )
                    
                        # tain and test model
                        df_metrics, df_metrics_best_model = model.fit(
                            train_preprocessed,
                            test_preprocessed
                        )
                        
                        # save model (for XAI assessment after)

                        # return metrics, metrics_best
                        experimental_results.append(df_metrics_best_model.to_numpy()[0])
                    
                
                    # save experimental results
                    experimental_results_df = pd.DataFrame(
                        columns=['best_model_train_loss', 'best_model_val_loss', 
                                'best_model_train_acc', 'best_model_val_acc', 
                                'best_model_learning_rate', 'best_model_nb_epoch'],
                        data=experimental_results
                    )
                    
                    experimental_results_df.to_csv(iter_experiments_name + "_experimental_results.csv")
        


if __name__ == '__main__':
    run_train_premodels_with_sourcedata()