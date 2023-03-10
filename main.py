import os
import sys
import time

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import shap
import lime
import lime.lime_image

from skimage.io import imsave, imread
from skimage.transform import resize

from models.explainer.gradcam import GradCAM
from utils.utils import load_img
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
TARGET_ITERATIONS = 5 # TODO: Reset from TEST -> 2
K_MAX = 26 # TODO: Reset from TEST -> 2 

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
                verbose = True
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
    
    pre_model_weights_path = BASE_PATH + "results/pre_models/weights/"
    weights_path = BASE_PATH + "results/experiments/models/"
    experiments_path = BASE_PATH + "results/experiments/results/"
    
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
                
                # data path for target
                target_data_path = TARGET_DATA_PATH + target_dataset_name + "/"
                
                # path to pre trained models
                model_source_weights_path = weights_path 

                
                for iteration in range(TARGET_ITERATIONS):
                    
                    # TODO: Parallelized @ this point -> Skip iteration 0
                    if iteration == 0:
                        continue
                    
                    print("######################################")
                    print(f"### Switching to iteration: {iteration} ###")
                    print("######################################")
                    
                    # TODO: Due to run <= 24 -> continue
                    if iteration <= 3:
                        continue
                    
                    # assembled name of experiment results per iteration per model per datasets
                    iter_experiments_name = experiments_path + "it_" + str(iteration) + "_" + \
                        premodel + "_" + source_dataset_name + "_" + target_dataset_name
                    
                    
                    # create list for later dataframe
                    experimental_results = []
                    
                    # load data set 
                    # w/ seed RANDOM_STATE + iteration 
                    train_ds = load_local_dataset_tf(
                        target_data_path, 
                        target_size=TARGET_SIZE, 
                        subset="training",
                        seed=RANDOM_STATE+iteration,
                        batch_size=1
                    )
                    
                    test_ds = load_local_dataset_tf(
                        target_data_path, 
                        target_size=TARGET_SIZE, 
                        subset="test",
                        seed=RANDOM_STATE+iteration,
                        batch_size=1
                    )
                    
                    test_size = int(test_ds.cardinality().numpy())
                    
                    train_preprocessed = preprocess_data_per_tfmodel(train_ds, model_name=premodel)
                    test_preprocessed = preprocess_data_per_tfmodel(test_ds, model_name=premodel)
                    
                
                    for k_shot in range(K_MAX):
                        
                        if k_shot == 0:
                            continue
                        # TODO: fix run
                        if k_shot < 24:
                            continue
                        print("######################################")
                        print(f"### Switching to k_shot: {k_shot} ###")
                        print("######################################")
                        
                        # reducing train to k_shot
                        k_shot_train_preprocessed = train_preprocessed.take(k_shot)
                        k_shot_test_preprocessed = test_preprocessed.take(test_size)
                        
                        print(f"IT IS {k_shot} SHOOTING")
                        print(type(k_shot))
                        print(k_shot_train_preprocessed.cardinality().numpy())
                        print()
                        print(f"And test size: {test_size}")
                        print()
                
                        
                        
                        # create model save path
                        k_shot_model_save_path = model_source_weights_path + \
                            "it_" + str(iteration) + "_" + premodel + "_" + source_dataset_name + \
                            "_" + target_dataset_name + "_kshot_" + str(k_shot) + "/"
                            
                        print("k_shot_model_save_path: " + k_shot_model_save_path)
                        print()
                        print()
                            
                        if not os.path.exists(k_shot_model_save_path):
                            os.makedirs(k_shot_model_save_path)
                        
                        # create full model
                        model = create_full_model(
                            weights_path,
                            premodel,
                            INPUT_SHAPE,
                            source_dataset_name,
                            target_dataset_name,
                            target_num_classes,
                            k_shot,
                            iteration,
                            pre_model_weights_path,
                            verbose=False # TODO: Reset from TEST -> False
                        )
                    
                        # tain and test model
                        df_metrics, df_metrics_best_model = model.fit(
                            k_shot_train_preprocessed,
                            k_shot_test_preprocessed
                        )
                    
                        # return metrics, metrics_best
                        experimental_results.append(df_metrics_best_model.to_numpy()[0])
                        
                        del k_shot_train_preprocessed, k_shot_test_preprocessed
                    
                
                    # save experimental results
                    experimental_results_df = pd.DataFrame(
                        columns=['best_model_train_loss', 'best_model_val_loss', 
                                'best_model_train_acc', 'best_model_val_acc', 
                                'best_model_learning_rate', 'best_model_nb_epoch'],
                        data=experimental_results
                    )
                    
                    experimental_results_df.to_csv(iter_experiments_name + "_experimental_results.csv")


def run_pretrained_fullmodels_sophisticated_evaluation():
    
    # generate acc, f1-score, kappa, AUC
    pass

        
def run_xai_evaluation_with_models():
    
    RESIZE = (275, 275)
    
    ## Load images 
    image_path = BASE_PATH + "data/target/mechanicalseals_fulllight/"
    class_names = ["damaged", "undamaged"] # as trained 
    
    
    #### Image 97 - damaged
    image97 = load_img(image_path + "damaged/" + "97.png", target_size=(224, 224))
    image97_resized = load_img(image_path + "damaged/" + "97.png", target_size=RESIZE)
    
    #### Image 71 - damaged
    image71 = load_img(image_path + "damaged/" + "71.png", target_size=(224, 224))
    image71_resized = load_img(image_path + "damaged/" + "71.png", target_size=RESIZE)
    
    #### Image 16 - undamaged
    image16 = load_img(image_path + "undamaged/" + "16.png", target_size=(224, 224))
    
    iterations = [
        0,
        1,
        2,
        3,
        4,
    ]  
    
    k_shot = [1, 5, 10, 15, 20, 25]
    
    models = [
        "vgg16", 
        "resnet101", 
        "densenet121"
    ]
    
    model_types = [
        "_model_last",
        "_model_best"
    ]
    
    source_datasets = [
        "imagenet",
        "dagm",
        "caltech101",
    ]
    
    target_datasets = [
        "mechanicalseals_fulllight",
    ]
    
    base_path = BASE_PATH + "results/experiments/models/"
    save_path = BASE_PATH + "results/xai/"
    
    
    
    for iteration in iterations:
        for model in models:
            
            preprocessing_func = None
            
            if model == "vgg16":
                preprocessing_func = tf.keras.applications.vgg16.preprocess_input
            elif model == "resnet101":
                preprocessing_func = tf.keras.applications.resnet.preprocess_input
            elif model == "densenet121":
                preprocessing_func = tf.keras.applications.densenet.preprocess_input
            
            image97_preprocessed = preprocessing_func(image97)
            image71_preprocessed = preprocessing_func(image71)
            image16_preprocessed = preprocessing_func(image16)
            
            for s_dataset in source_datasets:
                for t_dataset in target_datasets:
                    for k in k_shot:
                        
                        # print(iteration, " ", type(iteration))
                        # print(k, " ", type(k))
                        
                        model_path = "it_" + str(iteration) + "_" + model + "_" + s_dataset + "_" + t_dataset + "_kshot_" + str(k) 
                        path = base_path + model_path + "/"
                        
                        # print(path)
                        
                        # check if path exists
                        if os.path.exists(path) == False:
                            continue
                        
                        for model_type in model_types:
                        
                            # check if path exsists
                            modeltype_path = path + model_path + model_type + ".h5"
                            # print(best_path)
                            
                            if os.path.exists(modeltype_path) == False:
                                continue
                            
                            tf_model = tf.keras.models.load_model(modeltype_path)
                            print(f"model loaded: {model_path}{model_type}")
                            
                            pred_97 = np.argmax(tf_model.predict(image97_preprocessed))
                            pred_71 = np.argmax(tf_model.predict(image71_preprocessed))
                            
                            # print("97: ", class_names[pred_97])
                            # print("71: ", class_names[pred_71])
                            
                            
                            
                            ### GradCAM - Image 97
                            grad_cam_97 = GradCAM(tf_model, 0)
                            grad_cam_97_hm = grad_cam_97.compute_heatmap(image97_preprocessed)
                            grad_cam_97_res = grad_cam_97.overlay_heatmap(grad_cam_97_hm, image97[0])
                            grad_cam_97_img = Image.fromarray(grad_cam_97_res[1])
                            grad_cam_97_img = grad_cam_97_img.resize(RESIZE, Image.BICUBIC)
                            # grad_cam_97_img.save(save_path + model_path + "_damaged_gradcam_97.png")

                            ### GradCAM - Image 71
                            grad_cam_71 = GradCAM(tf_model, 0)
                            grad_cam_71_hm = grad_cam_71.compute_heatmap(image71_preprocessed)
                            grad_cam_71_res = grad_cam_71.overlay_heatmap(grad_cam_71_hm, image71[0])
                            grad_cam_71_img = Image.fromarray(grad_cam_71_res[1])
                            grad_cam_71_img = grad_cam_71_img.resize(RESIZE, Image.BICUBIC)
                            # grad_cam_71_img.save(save_path + model_path + "_damaged_gradcam_71.png")
                            
                            
                            ### LIME - Image 97
                            explainer = lime.lime_image.LimeImageExplainer()
                            
                            explanation = explainer.explain_instance(image97_preprocessed[0], 
                                tf_model.predict, 
                                # hide_color=(128, 128, 128),
                                hide_color=(0, 0, 0),
                                num_samples=5
                            )
                            
                            # Visualize the explanation
                            temp, mask = explanation.get_image_and_mask(
                                label=0, 
                                positive_only=False, 
                                negative_only=False, 
                                hide_rest=False,
                                num_features=20
                                )
                            
                            # Save the explanation as an image
                            lime_97_img = cv2.resize(temp, RESIZE, interpolation = cv2.INTER_CUBIC)
                            # cv2.imwrite(save_path + model_path + "_damaged_lime_97.png", temp)
                            
                            ### LIME - Image 71
                            explainer = lime.lime_image.LimeImageExplainer()
                            
                            explanation = explainer.explain_instance(image71_preprocessed[0], 
                                tf_model.predict, 
                                # hide_color=(128, 128, 128),
                                hide_color=(0, 0, 0),
                                num_samples=5
                            )
                            
                            # Visualize the explanation
                            temp, mask = explanation.get_image_and_mask(
                                label=0, 
                                positive_only=False, 
                                negative_only=False, 
                                hide_rest=False,
                                num_features=20
                                )
                            
                            # Save the explanation as an image
                            lime_71_img = cv2.resize(temp, RESIZE, interpolation = cv2.INTER_CUBIC)
                            # cv2.imwrite(save_path + model_path + "_damaged_lime_71.png", temp)
                            
                            
                            images97 = [image97_resized[0], grad_cam_97_img, lime_97_img]
                            images71 = [image71_resized[0], grad_cam_71_img, lime_71_img]
                            
                            # for image in images97:
                            #     print(np.array(image).shape)
                            
                            image97_comb = np.hstack((np.array(i) for i in images97))
                            image71_comb = np.hstack((np.array(i) for i in images71))
                            
                            print("image97_comb: ", type(image97_comb), image97_comb.shape)
                            
                            cv2.imwrite(save_path + model_path + "_damaged_97.png", image97_comb)
                            # image97_comb.save(save_path + model_path + "_damaged_97.png")
                            
                            cv2.imwrite(save_path + model_path + "_damaged_71.png", image71_comb)
                            # image71_comb.save(save_path + model_path + "_damaged_71.png")

                            
                            
                            # break
                        pass
                        # break
                    pass
                    # break
                pass
                # break
            pass
            # break
        pass
        # break
    pass
                            
                        
                        


def deduct_results():
    
    iterations = 5 # for range 
    k_shot = 50 # for range
    
    models = [
        "vgg16", 
        "resnet101", 
        "densenet121"
    ]
    
    source_datasets = [
        "imagenet",
        "dagm",
        "caltech101",
    ]
    
    target_datasets = [
        "mechanicalseals_fulllight",
    ]
    
    
    ## results for best and last (using last i guess)
    ##
    ## accuracy w/ standard deviation, 
    ##
    ## f1-score w/ standard deviation, kappa w/ standard deviation, AUC w/ standard deviation: nachtr??glich 
    ##    
    ## :::: accuracy per epoch; over all iteration per model //
    ## :::: accuracy per epoch; over all iteration over all models //
    ##
    ##
    ## :::: accuracy per k-shot; over all iteration per model 
    ## :::: loss per k-shot; over all iteration per model
    ## :::: accuracy per k-shot; over all iteration over all models
    ## :::: loss per k-shot; over all iteration over all models
    ##
    
    data_imagenet = {}
    data_dagm = {}
    
    last_cols = ["precision_train", "accuracy_train", "recall_train", "precision_test", "accuracy_test", "recall_test", "duration"]
    best_cols = ["best_model_train_loss", "best_model_val_loss", "best_model_train_acc", "best_model_val_acc", "best_model_learning_rate", "best_model_nb_epoch"]
    hist_cols = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]
    
    models_path =  BASE_PATH + "results/experiments/models/"

    identifier_cols = ["iteration", "model", "source_dataset", "target_dataset", "k_shot"]

    best_data_df = pd.DataFrame(columns=identifier_cols + best_cols)
    last_data_df = pd.DataFrame(columns=identifier_cols + last_cols)
    hist_data_df = pd.DataFrame(columns=identifier_cols + hist_cols)


    # print("best_data_df: " , best_data_df.columns)    
    # print("last_data_df: " , last_data_df.columns)    
    # print("hist_data_df: " , hist_data_df.columns) 
    for iteration in range(iterations):
        for model in models:
            for s_dataset in source_datasets:
                for t_dataset in target_datasets:
                    
                    for k in range(k_shot):
                        
                        if k == 0:
                            continue
                        
                        name = "it_" + str(iteration) + "_" + model + "_" + s_dataset + "_" + t_dataset + "_kshot_" + str(k)
                        
                        best_file_name = models_path + name + "/" + name + "_best_model.csv"
                        last_file_name = models_path + name + "/" + name + "_metrics.csv"
                        hist_file_name = models_path + name + "/" + name + "_history.csv"
                        
                        identifier_dict = {}
                        identifier_dict[identifier_cols[0]] = iteration
                        identifier_dict[identifier_cols[1]] = model
                        identifier_dict[identifier_cols[2]] = s_dataset
                        identifier_dict[identifier_cols[3]] = t_dataset
                        identifier_dict[identifier_cols[4]] = k
                        
                        if os.path.exists(best_file_name) == True:
                            best_df = pd.read_csv(best_file_name)
                            
                            best_dict = identifier_dict.copy()
                            best_dict.update(best_df.iloc[0].to_dict())
                            best_df = pd.DataFrame(best_dict, index=[0])
                            best_data_df = pd.concat([best_data_df, best_df], ignore_index=True)
                            
                        if os.path.exists(last_file_name) == True:
                            last_df = pd.read_csv(last_file_name)
                            
                            last_dict = identifier_dict.copy()
                            last_dict.update(last_df.iloc[0].to_dict())
                            last_df = pd.DataFrame(last_dict, index=[0])
                            last_data_df = pd.concat([last_data_df, last_df], ignore_index=True)

                        if os.path.exists(hist_file_name) == True:
                            hist_df = pd.read_csv(hist_file_name)
                            
                            for index, row in hist_df.iterrows():
                                hist_dict = identifier_dict.copy()
                                hist_dict.update(row.to_dict())
                                hist_dict.update({"epoch": index})
                                hist_df = pd.DataFrame(hist_dict, index=[0])
                                hist_data_df = pd.concat([hist_data_df, hist_df], ignore_index=True)
                        
                        
                        pass # end of k-shot for loop
                    pass # end of target dataset for loop
                pass # end of source dataset for loop
            pass # end of model for loop
        pass # end of iteration for loop
    
    
    
    print("best_data_df: ", best_data_df.shape)
    print("last_data_df: ", last_data_df.shape)
    print("hist_data_df: ", hist_data_df.shape)
    
    path = BASE_PATH + "results/experiments/"
    
    best_data_df.to_hdf(path + "best_data_df.h5", key="df", mode="w")
    last_data_df.to_hdf(path + "last_data_df.h5", key="df", mode="w")
    hist_data_df.to_hdf(path + "hist_data_df.h5", key="df", mode="w")
    
    

if __name__ == '__main__':
    # run_train_premodels_with_sourcedata()
    # run_train_models_with_targetdata()
    
    run_xai_evaluation_with_models()
    
    # deduct_results()