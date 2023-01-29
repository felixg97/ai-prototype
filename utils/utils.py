import glob
import numpy as np

import os
import cv2

from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split

from PIL import Image, ImageSequence, ImageDraw

"""
Dataset utils
"""

# load train test split w/ ability to persist state (dataset)
def load_local_dataset_train_test(path, img_format, test_size=.2, target_size=(350, 350), random_state=42):
    print(f"Loading data set w/ splits: {path}, test_size = {test_size}, target_size = {target_size}")

    X, y = load_local_dataset(path, img_format, target_size=target_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Train-Test-Split done: {X_train.shape[0]} train size, {X_test.shape[0]} test size)")
    return X_train, y_train, X_test, y_test


## read data from tif files
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
                X.append(np.array(seeked_image.resize(target_size), dtype=np.float64))
            else:
                X.append(np.array(image, dtype=np.float64))

            label = np.zeros(len(class_directories))
            label[class_dictionary[class_name]] = 1
            
            y.append(label)
        
    return np.array(X), np.array(y) 


## read data from png files # GERADE
def load_local_dataset(path, img_format, target_size=None, ignore_split=True):
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
                        X.append(np.array(image.resize(target_size), dtype=np.float64))
                    else:
                        X.append(np.array(image, dtype=np.float64))
                    
                    label = np.zeros(len(class_directories))
                    label[class_dictionary[class_name]] = 1
                    
                    y.append(label)
        
        else:
            for image_path in glob.glob(class_dir + "/*" + img_format):
                
                image = Image.open(image_path).convert("RGB")

                if target_size is not None:
                    X.append(np.array(image.resize(target_size), dtype=np.float64))
                else:
                    X.append(np.array(image, dtype=np.float64))
                
                label = np.zeros(len(class_directories))
                label[class_dictionary[class_name]] = 1
                
                y.append(label)
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    print(f"Dataset loaded: {X.shape[0]} images, {class_idx} classes (shape: {X.shape})")
    return X, y 



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
            #image = cv2.imread(image_path) #, -1)
            image = Image.open(image_path)#.convert('RGB') 
            
            cropImage = True
            
            #Iterate over different Images in tif file
            for i, page in enumerate(ImageSequence.Iterator(image)):
                #Use Image with complete Lighting
                if i == 4:
                    image = page
                if cropImage == True and i == 4:
            
                    height, width = image.size
                    
                    #Create Mask
                    lum_img = Image.new("L", [height, width], 255)
                    
                    #Lay Mask over irrelevant Areas
                    draw = ImageDraw.Draw(lum_img)
                    draw.pieslice([1100, 630, (2920, 2450)], 0, 360, fill=0, outline="red")

                    draw_inner_circle = ImageDraw.Draw(lum_img)
                    draw_inner_circle.pieslice([1470, 1010, (2540, 2070)], 0, 360, fill=255, outline="red")

                    image.paste(lum_img, (0,0),lum_img)
            
            #Convert to RGB
            image = image.convert('RGB')
            image = np.array(image, dtype=np.float64) 
            # Convert RGB to BGR 
            image = image[:, :, ::-1].copy() 
            
            # Resize
            border_v, border_h = 0, 0
            (IMG_ROW, IMG_COL) = target_size
            if (IMG_COL/IMG_ROW) >= (image.shape[0]/image.shape[1]):
                border_v = int((((IMG_COL/IMG_ROW)*image.shape[1])-image.shape[0])/2)
            else:
                border_h = int((((IMG_ROW/IMG_COL)*image.shape[0])-image.shape[1])/2)
            image = cv2.copyMakeBorder(image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            image = cv2.resize(image, (IMG_ROW, IMG_COL))
            
            
            # Transform data into target format
            image = image.astype(np.float64)
            #image = preprocess_input(image)
            
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
    
    print('Found', image_nparray.shape[0], 'images belonging to', class_idx, 'classes.')
    
    return image_nparray, label_nparray


"""
Evaluation utils
"""

# If models do not significantly change between each other this function will be reduced TODO
def create_premodel(model_name, dataset_name, input_shape, num_classes, path, build=True):
    if model_name == 'VGG16':
        from models.tf_models.VGG16_model import VGG16_model
        return VGG16_model(
            dataset_name,
            input_shape=input_shape,
            num_classes=num_classes,
            build_pre_model=build,
            path=path
        )
    elif model_name == 'ResNet101':
        from models.tf_models.ResNet101_model import ResNet101_model
        return ResNet101_model(
            dataset_name,
            input_shape=input_shape,
            num_classes=num_classes,
            build_pre_model=build,
            path=path
        )
    elif model_name == 'MobileNet':
        from models.tf_models.MobileNet_model import MobileNet_model
        return MobileNet_model(
            dataset_name,
            input_shape=input_shape,
            num_classes=num_classes,
            build_pre_model=build,
            path=path
        )
    elif model_name == 'DenseNet121':
        from models.tf_models.DenseNet121_model import DenseNet121_model
        return DenseNet121_model(
            dataset_name,
            input_shape=input_shape,
            num_classes=num_classes,
            build_pre_model=build,
            path=path
        )