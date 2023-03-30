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
import threading

# import shap
# import lime
# import lime.lime_image

# from skimage.io import imsave, imread
# from skimage.transform import resize

from models.explainer.gradcam import GradCAM
from utils.utils import load_img
from utils.utils import load_local_dataset_train_test, load_local_dataset_tf, preprocess_data_per_tfmodel, split_dataset_in_intact_and_defect_balanced
from utils.utils import create_premodel, create_full_model

from utils.constants import BASE_PATH
from utils.constants import SOURCE_DATASETS, TARGET_DATASETS
from utils.constants import TF_MODELS

source_data_path = BASE_PATH + "data/source/"
TARGET_DATA_PATH = BASE_PATH + "data/target/"

VERBOSE = True

# Run specific constants, other to find in utils/constants.py
# Overall stuff
RANDOM_STATE = 42
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (*TARGET_SIZE, 3)
BATCH_SIZE = 16

# Pre-model stuff
BUILD_PREMODEL = True

# Classification layer stuff
BUILD_MODEL = True
TARGET_ITERATIONS = 5  # TODO: Reset from TEST -> 2
K_MAX = 51  # TODO: Reset from TEST -> 2
K_MAX = 31  # TODO: Reset from TEST -> 2


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
                verbose=True
            )

            print(f"### Pre-model {premodel} instantiated. ###")
            print("######################################")

            train_preprocessed = None
            test_preprocessed = None

            if dataset_name != "imagenet":
                train_preprocessed = preprocess_data_per_tfmodel(
                    train_ds, model_name=premodel)
                test_preprocessed = preprocess_data_per_tfmodel(
                    test_ds, model_name=premodel)

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


def run_train_models_with_targetdata(multi_gpu=True):
    start = time.time()

    print("TensorFlow Version: ", tf.__version__)

    config = tf.compat.v1.ConfigProto()
    # allocate gpu memory as needed
    # config.gpu_options.allow_growth = True # method 1
    # allocate percentage of gpu memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # method 2
    # session
    # config.gpu_options.visible_device_list = "1"
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #
    session = tf.compat.v1.InteractiveSession(config=config)

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
                print(
                    f"### Switching to target dataset: {target_dataset_name} ###")
                print("######################################")

                # data path for target
                target_data_path = TARGET_DATA_PATH + target_dataset_name + "/"

                # path to pre trained models
                model_source_weights_path = weights_path

                if multi_gpu:

                    def train_top_model_iteration(
                        iteration,
                        weights_path,
                        pre_model_weights_path,
                        experiments_path,
                        premodel, source_dataset_name,
                        target_dataset_name,
                        target_num_classes,
                        gpu_devices,
                        gpu_tracker
                    ):
                        print("Started threaded training of top model")

                        gpu_device_name = gpu_tracker.pop()

                        gpu_device = None
                        for gpu in gpu_devices:
                            if gpu.name is gpu_device_name:
                                gpu_device = gpu

                        print(f"Iteration is running on {gpu_device}")

                        with gpu_device:

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

                            train_size = int(train_ds.cardinality().numpy())
                            test_size = int(test_ds.cardinality().numpy())

                            train_preprocessed = preprocess_data_per_tfmodel(
                                train_ds, model_name=premodel)
                            test_preprocessed = preprocess_data_per_tfmodel(
                                test_ds, model_name=premodel)

                            for k_shot in range(K_MAX):

                                if k_shot == 0:
                                    continue
                                # # TODO: fix run
                                if k_shot > 41:  # first run
                                    continue
                                if k_shot > 51:  # second run
                                    continue
                                print("######################################")
                                print(f"### Switching to k_shot: {k_shot} ###")
                                print("######################################")

                                # reduce to k_shot size 2(N)*k
                                k_shot_train_preprocessed = train_preprocessed.take(
                                    train_size)

                                # for img, label in k_shot_train_preprocessed.take(2):
                                #     print(img.shape, " ", label.numpy())

                                k_shot_train_preprocessed = split_dataset_in_intact_and_defect_balanced(
                                    k_shot_train_preprocessed, k_shot)

                                # for img, label in k_shot_train_preprocessed.take(2):
                                #     print(img.shape, " ", label.numpy())

                                # info: full test size
                                k_shot_test_preprocessed = test_preprocessed.take(
                                    test_size)

                                print(f"IT IS {k_shot} SHOOTING")
                                print(type(k_shot))
                                print(
                                    k_shot_train_preprocessed.cardinality().numpy())
                                print()
                                print(f"And test size: {test_size}")
                                print()

                                # create model save path
                                k_shot_model_save_path = model_source_weights_path + \
                                    "it_" + str(iteration) + "_" + premodel + "_" + source_dataset_name + \
                                    "_" + target_dataset_name + \
                                    "_kshot_" + str(k_shot) + "/"

                                print("k_shot_model_save_path: " +
                                      k_shot_model_save_path)
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
                                    verbose=False  # TODO: Reset from TEST -> False
                                )

                                # tain and test model
                                df_metrics, df_metrics_best_model = model.fit(
                                    k_shot_train_preprocessed,
                                    k_shot_test_preprocessed
                                )

                                # return metrics, metrics_best
                                experimental_results.append(
                                    df_metrics_best_model.to_numpy()[0])

                                del k_shot_train_preprocessed, k_shot_test_preprocessed

                            # save experimental results
                            experimental_results_df = pd.DataFrame(
                                columns=['best_model_train_loss', 'best_model_val_loss',
                                         'best_model_train_acc', 'best_model_val_acc',
                                         'best_model_learning_rate', 'best_model_nb_epoch'],
                                data=experimental_results
                            )

                            experimental_results_df.to_csv(
                                iter_experiments_name + "_experimental_results.csv")

                            print(f"Iteration ended on: {gpu_device}")

                        # end GPU usage
                        gpu_tracker.append(gpu_device_name)

                    def parallel_func(gpu_devices, gpu_tracker):
                        gpu_device_name = gpu_tracker.pop()

                        gpu_device = None
                        for gpu in gpu_devices:
                            if gpu.name is gpu_device_name:
                                gpu_device = gpu

                        print(f"Iteration is running on {gpu_device}")

                        print(gpu_tracker)

                        time.sleep(30)

                        gpu_tracker.append(gpu_device_name)

                    gpu_devices = tf.config.list_physical_devices("GPU")
                    gpu_tracker = [gpu.name for gpu in gpu_devices]

                    iterations = list(reversed([i for i in range(
                        TARGET_ITERATIONS)]))

                    while len(iterations) > 0:
                        if len(gpu_tracker) == 0:
                            print("Waiting for GPU")
                            time.sleep(10)
                        else:
                            iteration = iterations.pop()

                            print("######################################")
                            print(
                                f"### Switching to iteration: {iteration} ###")
                            print("######################################")

                            # thread = threading.Thread(
                            #     target=parallel_func,
                            #     args=(gpu_devices, gpu_tracker)
                            # )
                            # thread.start()

                            thread = threading.Thread(
                                target=train_top_model_iteration,
                                args=(
                                    iteration,
                                    weights_path,
                                    pre_model_weights_path,
                                    experiments_path,
                                    premodel, source_dataset_name,
                                    target_dataset_name,
                                    target_num_classes,
                                    gpu_devices,
                                    gpu_tracker
                                )
                            )
                            thread.start()

                else:
                    for iteration in range(TARGET_ITERATIONS):

                        # # TODO: Parallelized @ this point -> Skip iteration 0
                        # if iteration == 0:
                        #     continue

                        # # TODO: Due to run <= 24 -> continue
                        # if iteration <= 3:
                        #     continue

                        # if premodel == "vgg16" \
                        #         and source_dataset_name == "imagenet":
                        #     continue

                        # if premodel == "resnet101" \
                        #     and source_dataset_name == "caltech101" \
                        #         and iteration == 4:
                        #     pass
                        # elif premodel == "vgg16" \
                        #     and source_dataset_name == "imagenet" \
                        #         and 2 <= iteration and iteration <= 4:
                        #     pass
                        # elif premodel == "vgg16" \
                        #     and source_dataset_name == "imagenet" \
                        #         and iteration == 0:
                        #     pass
                        # else:
                        #     continue

                        if premodel == "vgg16" \
                            and source_dataset_name == "caltech101":
                            pass
                        elif premodel == "vgg16" \
                            and source_dataset_name == "dagm":
                            pass
                        else:
                            continue


                        print("######################################")
                        print(f"### Switching to iteration: {iteration} ###")
                        print("######################################")

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

                        train_size = int(train_ds.cardinality().numpy())
                        test_size = int(test_ds.cardinality().numpy())

                        train_preprocessed = preprocess_data_per_tfmodel(
                            train_ds, model_name=premodel)
                        test_preprocessed = preprocess_data_per_tfmodel(
                            test_ds, model_name=premodel)

                        for k_shot in range(K_MAX):

                            if k_shot == 0:
                                continue
                            # # TODO: fix run
                            # if k_shot < 31:  # first run, adapted run from 41 to 31
                            #     continue
                            if k_shot > 51:  # second run
                                continue
                            print("######################################")
                            print(f"### Switching to k_shot: {k_shot} ###")
                            print("######################################")

                            # reduce to k_shot size 2(N)*k
                            k_shot_train_preprocessed = train_preprocessed.take(
                                train_size)

                            # for img, label in k_shot_train_preprocessed.take(2):
                            #     print(img.shape, " ", label.numpy())

                            k_shot_train_preprocessed = split_dataset_in_intact_and_defect_balanced(
                                k_shot_train_preprocessed, k_shot)

                            # for img, label in k_shot_train_preprocessed.take(2):
                            #     print(img.shape, " ", label.numpy())

                            # info: full test size
                            k_shot_test_preprocessed = test_preprocessed.take(
                                test_size)

                            print(f"IT IS {k_shot} SHOOTING")
                            print(type(k_shot))
                            print(k_shot_train_preprocessed.cardinality().numpy())
                            print()
                            print(f"And test size: {test_size}")
                            print()

                            # create model save path
                            k_shot_model_save_path = model_source_weights_path + \
                                "it_" + str(iteration) + "_" + premodel + "_" + source_dataset_name + \
                                "_" + target_dataset_name + \
                                "_kshot_" + str(k_shot) + "/"

                            print("k_shot_model_save_path: " +
                                  k_shot_model_save_path)
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
                                verbose=False  # TODO: Reset from TEST -> False
                            )

                            # tain and test model
                            df_metrics, df_metrics_best_model = model.fit(
                                k_shot_train_preprocessed,
                                k_shot_test_preprocessed
                            )

                            # return metrics, metrics_best
                            experimental_results.append(
                                df_metrics_best_model.to_numpy()[0])

                            del k_shot_train_preprocessed, k_shot_test_preprocessed

                        # save experimental results
                        experimental_results_df = pd.DataFrame(
                            columns=['best_model_train_loss', 'best_model_val_loss',
                                     'best_model_train_acc', 'best_model_val_acc',
                                     'best_model_learning_rate', 'best_model_nb_epoch'],
                            data=experimental_results
                        )

                        experimental_results_df.to_csv(
                            iter_experiments_name + "_experimental_results.csv")


def run_pretrained_fullmodels_sophisticated_evaluation():

    # generate acc, f1-score, kappa, AUC
    pass


def run_xai_evaluation_with_models():

    RESIZE = (275, 275)

    # Load images
    image_path = BASE_PATH + "data/target/mechanicalseals_fulllight/"
    class_names = ["damaged", "undamaged"]  # as trained

    # Image 97 - damaged
    image97 = load_img(image_path + "damaged/" +
                       "97.png", target_size=(224, 224))
    image97_resized = load_img(
        image_path + "damaged/" + "97.png", target_size=RESIZE)

    # Image 71 - damaged
    image71 = load_img(image_path + "damaged/" +
                       "71.png", target_size=(224, 224))
    image71_resized = load_img(
        image_path + "damaged/" + "71.png", target_size=RESIZE)

    # Image 16 - undamaged
    image16 = load_img(image_path + "undamaged/" +
                       "16.png", target_size=(224, 224))

    iterations = [
        0,
        1,
        2,
        3,
        4,
    ]

    k_shot = [1,2,3,4,5, 6,7,8,9,10, 15, 20, 25]

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

                        model_path = "it_" + \
                            str(iteration) + "_" + model + "_" + \
                            s_dataset + "_" + t_dataset + "_kshot_" + str(k)
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

                            tf_model = tf.keras.models.load_model(
                                modeltype_path)
                            print(f"model loaded: {model_path}{model_type}")

                            pred_97 = np.argmax(
                                tf_model.predict(image97_preprocessed))
                            pred_71 = np.argmax(
                                tf_model.predict(image71_preprocessed))

                            # print("97: ", class_names[pred_97])
                            # print("71: ", class_names[pred_71])

                            # GradCAM - Image 97
                            grad_cam_97 = GradCAM(tf_model, 0)
                            grad_cam_97_hm = grad_cam_97.compute_heatmap(
                                image97_preprocessed)
                            grad_cam_97_res = grad_cam_97.overlay_heatmap(
                                grad_cam_97_hm, image97[0])
                            grad_cam_97_img = Image.fromarray(
                                grad_cam_97_res[1])
                            grad_cam_97_img = grad_cam_97_img.resize(
                                RESIZE, Image.BICUBIC)
                            # grad_cam_97_img.save(save_path + model_path + "_damaged_gradcam_97.png")

                            # GradCAM - Image 71
                            grad_cam_71 = GradCAM(tf_model, 0)
                            grad_cam_71_hm = grad_cam_71.compute_heatmap(
                                image71_preprocessed)
                            grad_cam_71_res = grad_cam_71.overlay_heatmap(
                                grad_cam_71_hm, image71[0])
                            grad_cam_71_img = Image.fromarray(
                                grad_cam_71_res[1])
                            grad_cam_71_img = grad_cam_71_img.resize(
                                RESIZE, Image.BICUBIC)
                            # grad_cam_71_img.save(save_path + model_path + "_damaged_gradcam_71.png")

                            # LIME - Image 97
                            # explainer = lime.lime_image.LimeImageExplainer()

                            # explanation = explainer.explain_instance(image97_preprocessed[0],
                            #                                          tf_model.predict,
                            #                                          # hide_color=(128, 128, 128),
                            #                                          hide_color=(
                            #                                              0, 0, 0),
                            #                                          num_samples=5
                            #                                          )

                            # # Visualize the explanation
                            # temp, mask = explanation.get_image_and_mask(
                            #     label=0,
                            #     positive_only=False,
                            #     negative_only=False,
                            #     hide_rest=False,
                            #     num_features=20
                            # )

                            # # Save the explanation as an image
                            # lime_97_img = cv2.resize(
                            #     temp, RESIZE, interpolation=cv2.INTER_CUBIC)
                            # # cv2.imwrite(save_path + model_path + "_damaged_lime_97.png", temp)

                            # # LIME - Image 71
                            # explainer = lime.lime_image.LimeImageExplainer()

                            # explanation = explainer.explain_instance(image71_preprocessed[0],
                            #                                          tf_model.predict,
                            #                                          # hide_color=(128, 128, 128),
                            #                                          hide_color=(
                            #                                              0, 0, 0),
                            #                                          num_samples=5
                            #                                          )

                            # # Visualize the explanation
                            # temp, mask = explanation.get_image_and_mask(
                            #     label=0,
                            #     positive_only=False,
                            #     negative_only=False,
                            #     hide_rest=False,
                            #     num_features=20
                            # )

                            # # Save the explanation as an image
                            # lime_71_img = cv2.resize(
                            #     temp, RESIZE, interpolation=cv2.INTER_CUBIC)
                            # # cv2.imwrite(save_path + model_path + "_damaged_lime_71.png", temp)

                            images97 = [image97_resized[0],
                                        grad_cam_97_img]
                            images71 = [image71_resized[0],
                                        grad_cam_71_img]

                            # for image in images97:
                            #     print(np.array(image).shape)

                            image97_comb = np.hstack(
                                (np.array(i) for i in images97))
                            image71_comb = np.hstack(
                                (np.array(i) for i in images71))

                            print("image97_comb: ", type(
                                image97_comb), image97_comb.shape)

                            ### 97
                            cv2.imwrite(save_path + model_path +
                                        f"_damaged_97_{pred_97}{model_type}.png", image97_comb)

                            pd.DataFrame(grad_cam_97_hm)\
                                .to_csv(save_path + model_path + f"_damaged_97_{pred_97}{model_type}.csv", index=False)

                            ### 71
                            cv2.imwrite(save_path + model_path +
                                        f"_damaged_71_{pred_71}{model_type}.png", image71_comb)

                            pd.DataFrame(grad_cam_71_hm)\
                                .to_csv(save_path + model_path + f"_damaged_71_{pred_71}{model_type}.csv", index=False)

 
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

    iterations = 5  # for range
    k_shot = 50  # for range

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

    # results for best and last (using last i guess)
    ##
    # accuracy w/ standard deviation,
    ##
    # f1-score w/ standard deviation, kappa w/ standard deviation, AUC w/ standard deviation: nachträglich
    ##
    # :::: accuracy per epoch; over all iteration per model //
    # :::: accuracy per epoch; over all iteration over all models //
    ##
    ##
    # :::: accuracy per k-shot; over all iteration per model
    # :::: loss per k-shot; over all iteration per model
    # :::: accuracy per k-shot; over all iteration over all models
    # :::: loss per k-shot; over all iteration over all models
    ##

    data_imagenet = {}
    data_dagm = {}

    # last_cols = ["precision_train", "accuracy_train", "recall_train",
    #              "precision_test", "accuracy_test", "recall_test", "duration"]
    # best_cols = ["best_model_train_loss", "best_model_val_loss", "best_model_train_acc",
    #              "best_model_val_acc", "best_model_learning_rate", "best_model_nb_epoch"]
    # hist_cols = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]

    best_cols = ['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
       'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch']
    last_cols = ['precision_train', 'accuracy_train', 'recall_train', 'precision_test',
       'accuracy_test', 'recall_test', 'duration']
    hist_cols = ['loss', 'accuracy', 'auc', 'true_negatives', 'true_positives', 'false_negatives', 
        'false_positives', 'val_loss', 'val_accuracy', 'val_auc', 'val_true_negatives', 
        'val_true_positives', 'val_false_negatives', 'val_false_positives']

    models_path = BASE_PATH + "results/experiments/models/"

    identifier_cols = ["iteration", "model",
                       "source_dataset", "target_dataset", "k_shot"]

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


                        name = "it_" + str(iteration) + "_" + model + "_" + \
                            s_dataset + "_" + t_dataset + "_kshot_" + str(k)

                        best_file_name = models_path + name + "/" + name + "_best_model.csv"
                        last_file_name = models_path + name + "/" + name + "_metrics.csv"
                        hist_file_name = models_path + name + "/" + name + "_history.csv"

                        identifier_dict = {}
                        identifier_dict[identifier_cols[0]] = iteration
                        identifier_dict[identifier_cols[1]] = model
                        identifier_dict[identifier_cols[2]] = s_dataset
                        identifier_dict[identifier_cols[3]] = t_dataset
                        identifier_dict[identifier_cols[4]] = k

                        TESTING = False

                        if os.path.exists(best_file_name) == True:
                            best_df = pd.read_csv(best_file_name)
                            if TESTING:
                                print("best: ", best_df.columns)
                            if not TESTING:
                                best_dict = identifier_dict.copy()
                                best_dict.update(best_df.iloc[0].to_dict())
                                best_df = pd.DataFrame(best_dict, index=[0])
                                best_data_df = pd.concat(
                                    [best_data_df, best_df], ignore_index=True)

                        if os.path.exists(last_file_name) == True:
                            last_df = pd.read_csv(last_file_name)
                            if TESTING:
                                print("last: ", last_df.columns)
                            if not TESTING:
                                last_dict = identifier_dict.copy()
                                last_dict.update(last_df.iloc[0].to_dict())
                                last_df = pd.DataFrame(last_dict, index=[0])
                                last_data_df = pd.concat(
                                    [last_data_df, last_df], ignore_index=True)

                        if os.path.exists(hist_file_name) == True:
                            hist_df = pd.read_csv(hist_file_name)
                            if TESTING:
                                print("hist: ", hist_df.columns)
                            if not TESTING:
                                for index, row in hist_df.iterrows():
                                    hist_dict = identifier_dict.copy()
                                    hist_dict.update(row.to_dict())
                                    hist_dict.update({"epoch": index})
                                    hist_df = pd.DataFrame(hist_dict, index=[0])
                                    hist_data_df = pd.concat(
                                        [hist_data_df, hist_df], ignore_index=True)

                        pass  # end of k-shot for loop
                    pass  # end of target dataset for loop
                pass  # end of source dataset for loop
            pass  # end of model for loop
        pass  # end of iteration for loop

    print("best_data_df: ", best_data_df.shape)
    print("last_data_df: ", last_data_df.shape)
    print("hist_data_df: ", hist_data_df.shape)

    path = BASE_PATH + "results/experiments/"

    best_data_df.to_hdf(path + "best_data_df.h5", key="df", mode="w")
    last_data_df.to_hdf(path + "last_data_df.h5", key="df", mode="w")
    hist_data_df.to_hdf(path + "hist_data_df.h5", key="df", mode="w")


def test_some():

    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        print(gpu)


if __name__ == '__main__':
    # run_train_premodels_with_sourcedata()
    # run_train_models_with_targetdata(multi_gpu=False)

    # run_xai_evaluation_with_models()

    deduct_results()

    # test_some()
