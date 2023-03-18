import glob
import os
import cv2

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.data as tfdata
import matplotlib.pyplot as plt

import tensorflow.keras as keras
# should be in the model itself
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from PIL import Image, ImageSequence, ImageDraw

from utils.constants import BASE_PATH

"""
Dataset utils
"""

# load train test split w/ ability to persist state (dataset)


def load_local_dataset_train_test(path, img_format, test_size=.2, target_size=(350, 350), random_state=42):
    print(
        f"Loading data set w/ splits: {path}, test_size = {test_size}, target_size = {target_size}")

    X, y = load_local_dataset(path, img_format, target_size=target_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    print(
        f"Train-Test-Split done: {X_train.shape[0]} train size, {X_test.shape[0]} test size)")
    return X_train, y_train, X_test, y_test


# read data from tif files
def load_local_dataset_from_tif(path, num_tif=4, target_size=(350, 350)):
    X, y = [], []

    class_dictionary = {}
    class_idx = 0

    class_directories = glob.glob(path + "/*", recursive=True)

    # open tif files
    for class_dir in class_directories:
        class_name = class_dir.split("/")[-1]
        class_dictionary[class_name] = class_idx
        class_idx += 1

        for image_path in glob.glob(class_dir + "/*.tif"):

            image = Image.open(image_path)

            image.seek(num_tif)
            seeked_image = image.copy().convert("RGB")

            if target_size is not None:
                X.append(np.array(seeked_image.resize(
                    target_size), dtype=np.float64))
            else:
                X.append(np.array(image, dtype=np.float64))

            label = np.zeros(len(class_directories))
            label[class_dictionary[class_name]] = 1

            y.append(label)

    return np.array(X), np.array(y)


# read data from png files # GOTO
def load_local_dataset(
    path, img_format,
    target_size=None,
    ignore_split=True,
    interpolation="inter_area",
):
    X, y = [], []

    class_dictionary = {}
    class_idx = 0

    class_directories = glob.glob(path + "/*", recursive=True)

    for class_dir in class_directories:
        class_name = class_dir.split("/")[-1]
        class_dictionary[class_name] = class_idx
        class_idx += 1

        sub_class_dirs = glob.glob(class_dir + "/*")

        if len(sub_class_dirs) == 2:
            # print("C'est un train test split ici")

            for sub_class_dir in sub_class_dirs:
                image_dir = glob.glob(sub_class_dir + "/*" + img_format)

                # print(sub_class_dir, "/*" + img_format, len(image_dir))

                for image_path in glob.glob(sub_class_dir + "/*" + img_format):

                    image = Image.open(image_path).convert("RGB")

                    if target_size is not None:
                        X.append(np.array(image.resize(
                            target_size), dtype=np.float64))
                    else:
                        X.append(np.array(image, dtype=np.float64))

                    label = np.zeros(len(class_directories))
                    label[class_dictionary[class_name]] = 1

                    y.append(label)

        else:
            for image_path in glob.glob(class_dir + "/*" + img_format):

                image = Image.open(image_path).convert("RGB")

                if target_size is not None:
                    X.append(np.array(image.resize(
                        target_size), dtype=np.float64))
                else:
                    X.append(np.array(image, dtype=np.float64))

                label = np.zeros(len(class_directories))
                label[class_dictionary[class_name]] = 1

                y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)

    print(
        f"Dataset loaded: {X.shape[0]} images, {class_idx} classes (shape: {X.shape})")
    return X, y


# read data from all files using tensorflow # GOTO
def load_local_dataset_tf(
    path,
    test_size=.2,
    target_size=None,
    batch_size=None,
    seed=42,
    subset=None,
    colormode="rgb",
    label_mode="categorical",
    interpolation="area",
):

    # ausgelagert in utils, da area die/das beste interpolation/resampling ist
    # is äquivalent zu Pillow Image.resize
    data = None

    if subset == "training":
        data = keras.utils.image_dataset_from_directory(
            path,
            validation_split=test_size,
            subset="training",
            seed=seed,
            image_size=target_size,
            batch_size=batch_size,
            interpolation=interpolation,
            color_mode=colormode,
            label_mode=label_mode,
            labels="inferred",
        )
    elif subset == "test":
        data = keras.utils.image_dataset_from_directory(
            path,
            validation_split=test_size,
            subset="validation",
            seed=seed,
            image_size=target_size,
            batch_size=batch_size,
            interpolation=interpolation,
            color_mode=colormode,
            label_mode=label_mode,
            labels="inferred",
        )
    else:
        data = keras.utils.image_dataset_from_directory(
            path,
            validation_split=test_size,
            subset=subset,
            seed=seed,
            image_size=target_size,
            batch_size=batch_size,
            interpolation=interpolation,
            color_mode=colormode,
            label_mode=label_mode,
            labels="inferred",
        )

    return data


# preprocess data
def preprocess_data_per_tfmodel(dataset, model_name="vgg16"):

    preprocessed = dataset

    # tune performance of tf.data.Dataset
    AUTOTUNE = tfdata.experimental.AUTOTUNE
    preprocessed = preprocessed.cache().prefetch(buffer_size=AUTOTUNE)

    if model_name == "vgg16":
        preprocessed = dataset.map(lambda x, y: (
            keras.applications.vgg16.preprocess_input(x), y))
    elif model_name == "resnet101":
        preprocessed = dataset.map(lambda x, y: (
            keras.applications.resnet50.preprocess_input(x), y))
    elif model_name == "densenet121":
        preprocessed = dataset.map(lambda x, y: (
            keras.applications.densenet.preprocess_input(x), y))
    else:
        raise Exception("Model not found")

    return preprocessed


# Helper function to preload images from directory
def preload_from_directory_old(directory_path, target_size=(350, 350)):
    image_list = []
    label_list = []
    class_dictionary = {}
    files_list = []
    class_idx = 0

    # Folder names equal the class names
    # Directories are read in one after the other
    class_directories = glob.glob(directory_path + '/*', recursive=True)
    for class_directory in class_directories:

        # Name and number for each class
        class_name = os.path.basename(class_directory)
        class_dictionary[class_name] = class_idx
        class_idx += 1

        # Read in images for each class
        image_files = glob.glob(class_directory + '/*.png')
        for image_path in image_files:
            # Pre-Process images for training
            # image = cv2.imread(image_path) #, -1)
            image = Image.open(image_path)  # .convert('RGB')

            cropImage = True

            # Iterate over different Images in tif file
            for i, page in enumerate(ImageSequence.Iterator(image)):
                # Use Image with complete Lighting
                if i == 4:
                    image = page
                if cropImage == True and i == 4:

                    height, width = image.size

                    # Create Mask
                    lum_img = Image.new("L", [height, width], 255)

                    # Lay Mask over irrelevant Areas
                    draw = ImageDraw.Draw(lum_img)
                    draw.pieslice([1100, 630, (2920, 2450)], 0,
                                  360, fill=0, outline="red")

                    draw_inner_circle = ImageDraw.Draw(lum_img)
                    draw_inner_circle.pieslice(
                        [1470, 1010, (2540, 2070)], 0, 360, fill=255, outline="red")

                    image.paste(lum_img, (0, 0), lum_img)

            # Convert to RGB
            image = image.convert('RGB')
            image = np.array(image, dtype=np.float64)
            # Convert RGB to BGR
            image = image[:, :, ::-1].copy()

            # Resize
            border_v, border_h = 0, 0
            (IMG_ROW, IMG_COL) = target_size
            if (IMG_COL/IMG_ROW) >= (image.shape[0]/image.shape[1]):
                border_v = int(
                    (((IMG_COL/IMG_ROW)*image.shape[1])-image.shape[0])/2)
            else:
                border_h = int(
                    (((IMG_ROW/IMG_COL)*image.shape[0])-image.shape[1])/2)
            image = cv2.copyMakeBorder(
                image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            image = cv2.resize(image, (IMG_ROW, IMG_COL))

            # Transform data into target format
            image = image.astype(np.float64)
            # image = preprocess_input(image)

            # Create label with one-hot encoding
            label = np.zeros(len(class_directories))
            label[class_dictionary[class_name]] = 1

            # Create list
            image_list.append(image)
            label_list.append(label)
            files_list.append(os.path.basename(image_path))

    # Convert list to array
    image_nparray = np.asarray(image_list)
    label_nparray = np.asarray(label_list)
    del image_list, label_list

    print('Found', image_nparray.shape[0],
          'images belonging to', class_idx, 'classes.')

    return image_nparray, label_nparray


def split_dataset_in_intact_and_defect_balanced(dataset, k):

    # Collect images and labels of the desired class
    images_defect = []
    labels_defect = []
    images_intact = []
    labels_intact = []

    i = 0
    for image, label in dataset.unbatch():
        if label.numpy()[0] == 0:  # intact
            images_intact.append(image.numpy())
            labels_intact.append(label.numpy())
        elif label.numpy()[0] == 1:  # defect
            images_defect.append(image.numpy())
            labels_defect.append(label.numpy())

        i += 1
        # if i == 3:
        #     break

    images_defect = np.array(images_defect[:k])
    labels_defect = np.array(labels_defect[:k])
    defect = tf.data.Dataset.from_tensor_slices(
        (images_defect, labels_defect))

    images_intact = np.array(images_intact[:k])
    labels_intact = np.array(labels_intact[:k])
    intact = tf.data.Dataset.from_tensor_slices(
        (images_intact, labels_intact))

    print()
    print("defect: ", len(images_defect))
    print("intact: ", len(images_intact))
    print()

    images = np.concatenate((images_defect, images_intact))
    labels = np.concatenate((labels_defect, labels_intact))

    result = tf.data.Dataset.from_tensor_slices((images, labels))

    return result


"""
Evaluation utils
"""

# If models do not significantly change between each other this function will be reduced TODO


def create_premodel(path, model_name, input_shape, dataset_name, num_classes, verbose=False):

    if model_name == 'vgg16':
        from models.tf_models.VGG16_model import VGG16_model
        return VGG16_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=True,
            source_data_name=dataset_name,
            source_num_classes=num_classes,
            verbose=verbose
        )
    elif model_name == 'resnet101':
        from models.tf_models.ResNet101_model import ResNet101_model
        return ResNet101_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=True,
            source_data_name=dataset_name,
            source_num_classes=num_classes,
            verbose=verbose
        )
    elif model_name == 'mobilenet':
        from models.tf_models.MobileNet_model import MobileNet_model
        return MobileNet_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=True,
            source_data_name=dataset_name,
            source_num_classes=num_classes,
            verbose=verbose
        )
    elif model_name == 'densenet121':
        from models.tf_models.DenseNet121_model import DenseNet121_model
        return DenseNet121_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=True,
            source_data_name=dataset_name,
            source_num_classes=num_classes,
            verbose=verbose
        )


def create_full_model(path, model_name, input_shape, source_dataset_name,
                      target_dataset_name, num_classes, k_shot, iteration, pre_model_path, verbose=False):

    if model_name == 'vgg16':
        from models.tf_models.VGG16_model import VGG16_model
        return VGG16_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=False,
            source_data_name=source_dataset_name,
            build_top_model=True,
            target_data_name=target_dataset_name,
            target_num_classes=num_classes,
            k_shot=k_shot,
            iteration=iteration,
            pre_model_path=pre_model_path,
            verbose=verbose
        )
    elif model_name == 'resnet101':
        from models.tf_models.ResNet101_model import ResNet101_model
        return ResNet101_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=False,
            source_data_name=source_dataset_name,
            build_top_model=True,
            target_data_name=target_dataset_name,
            target_num_classes=num_classes,
            k_shot=k_shot,
            iteration=iteration,
            pre_model_path=pre_model_path,
            verbose=verbose
        )
    elif model_name == 'mobilenet':
        from models.tf_models.MobileNet_model import MobileNet_model
        return MobileNet_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=False,
            source_data_name=source_dataset_name,
            build_top_model=True,
            target_data_name=target_dataset_name,
            target_num_classes=num_classes,
            k_shot=k_shot,
            iteration=iteration,
            pre_model_path=pre_model_path,
            verbose=verbose
        )
    elif model_name == 'densenet121':
        from models.tf_models.DenseNet121_model import DenseNet121_model
        return DenseNet121_model(
            path=path,
            input_shape=input_shape,
            build_pre_model=False,
            source_data_name=source_dataset_name,
            build_top_model=True,
            target_data_name=target_dataset_name,
            target_num_classes=num_classes,
            k_shot=k_shot,
            iteration=iteration,
            pre_model_path=pre_model_path,
            verbose=verbose
        )


def save_logs(path, hist, y_train, y_pred_train, y_test, y_pred_test, duration, save=True):

    hist_df = pd.DataFrame(hist.history)
    if save:
        hist_df.to_csv(path + '_history.csv', index=False)

    df_metrics = calculate_metrics(
        y_train, y_pred_train, y_test, y_pred_test, duration,)
    if save:
        df_metrics.to_csv(path + '_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']

    if save:
        df_best_model.to_csv(path + '_best_model.csv', index=False)
    plot_epochs_metric(hist, path + '_epochs_loss.png', metric='loss')

    return df_metrics, df_best_model


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, duration,):

    # print("y_train: ", y_train.shape)
    # print("y_pred_train: ", y_pred_train.shape)
    # print("y_test: ", y_test.shape)
    # print("y_pred_test: ", y_pred_test.shape)

    res = pd.DataFrame(data=np.zeros((1, 7), dtype=np.float), index=[0],
                       columns=['precision_train', 'accuracy_train', 'recall_train',
                                'precision_test', 'accuracy_test', 'recall_test', 'duration'])

    res['precision_train'] = precision_score(
        y_train, y_pred_train, average='macro')
    res['accuracy_train'] = accuracy_score(y_train, y_pred_train)
    res['recall_train'] = recall_score(y_train, y_pred_train, average='macro')
    res['precision_test'] = precision_score(
        y_test, y_pred_test, average='macro')
    res['accuracy_test'] = accuracy_score(y_test, y_pred_test)
    res['recall_test'] = recall_score(y_test, y_pred_test, average='macro')
    res['duration'] = duration

    return res


def load_img(path, target_size=None):

    img = Image.open(path)
    img = img.convert('RGB')

    if target_size:
        img = img.resize((target_size))
        img = np.array(img)

    img = np.expand_dims(img, axis=0)

    return img
