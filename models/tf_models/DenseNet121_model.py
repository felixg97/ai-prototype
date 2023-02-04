
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import time

from keras.models import load_model
from utils.constants import BASE_PATH
from utils.utils import save_logs



class DenseNet121_model():
    
    def __init__(self, 
                path = BASE_PATH,
                input_shape = None,
                build_pre_model = False, # source begin
                source_data_name = None, 
                source_num_classes = None, # source end 
                build_top_model = False, # target begin
                trainable_pre_model = False,
                target_data_name = None,
                target_num_classes = None, 
                k_shot = None, # target end
                verbose = False
            ):
        
        self.model_name = "densenet121"
        self.verbose = verbose
        
        self.model = None
        self.input_shape = input_shape
        self.cross_entropy = "categorical_crossentropy"
        
        ## source stuff
        self.build_pre_model_flag = build_pre_model
        self.trainable_pre_model_flag = trainable_pre_model
        self.source_data_name = source_data_name
        self.source_num_classes = source_num_classes
        self.pre_model_file_name = self.model_name + "_" \
            + self.source_data_name
        self.pre_trained_pre_model_path = path + self.pre_model_file_name
        
        ## target stuff
        self.build_top_model_flag = build_top_model
        self.target_data_name = target_data_name
        self.target_num_classes = target_num_classes
            
        if build_pre_model == True:
            self.build_pre_model()
            if verbose: 
                self.model.summary()
            return 
        else:
            weights_path = self.top_trained_pre_model_path + self.top_model_file_name
            self.model = load_model(weights_path + "_model_best.h5")
        
        
        ## down here bc of sequential processing 
        self.top_model_file_name = self.model_name + "_" \
            + self.source_data_name + "_" + self.target_data_name + "_" + str(k_shot)
        self.top_trained_pre_model_path = path + self.top_model_file_name
            
        if build_top_model == True:
            self.build_top_model()
            if verbose: 
                self.model.summary()
        else: 
            weights_path = self.top_trained_pre_model_path + self.top_model_file_name
            self.model = load_model(weights_path + "_model_best.h5")
        
    
    def build_pre_model(self):
        self.model = None # TF model
        
        if self.source_data_name == "imagenet":
            
            self.model = keras.applications.DenseNet121(
                input_shape=self.input_shape,
                weights="imagenet",
                include_top=False
            )
        
        self.model = keras.applications.DenseNet121(
            weights=None,
            input_shape=self.input_shape,
            classes=self.source_num_classes,
            include_top=True
        )
        
        # Usa RMSprop optimizer w/ lr=1e-4 
        optimizer = keras.optimizers.RMSprop(learning_rate=1e-4)
        
        self.model.compile(
            loss=self.cross_entropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', 
        #     factor=0.5, patience=200, min_lr=0.1)
        
        file_path = self.pre_trained_pre_model_path + \
            self.pre_model_file_name + "_model_best.h5"

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, 
            monitor='loss', 
            save_best_only=True
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
        )

        self.callbacks = [
            # reduce_lr, 
            model_checkpoint,
            early_stopping,
        ]
        
        if self.verbose == True:
            # Show model summary
            self.model.summary()
            
            
    def build_top_model(self):
        
        # load pretrained model - loaded
        if self.build_pre_model_flag is False:
            raise Exception("You must set buil pre model to true")

        if self.trainable_pre_model_flag is False:
            # freeze all layers
            for layer in self.model.layers:
                layer.trainable = False
        
        
        # Classification block
        x = None
        
        if self.source_data_name == "imagenet":
            x = self.model.layers[-1].ouput 
        else:
            x = self.model.layers[-4].output # test if it is the right layer TODO:
            
        x = keras.layers.Flatten(name="flatten")(x)
        x = keras.layers.Dense(32, activation="relu", name="fc1")(x)
        x = keras.layers.Dense(32, activation="relu", name="fc2")(x)
        
        output_layer = keras.layers.Dense(
            self.target_num_classes, activation="softmax", name="predictions")(x)
        
        self.model = keras.models.Model(
            input=self.model.input, 
            outputs=output_layer)
        
        # Usa RMSprop optimizer
        optimizer = keras.optimizers.RMSprop(learning_rate=1e-5)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer, 
            loss=self.cross_entropy, 
            metrics=['accuracy']
        )
        
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', 
        #     factor=0.5, patience=200, min_lr=0.1)

        file_path = self.top_trained_pre_model_path + \
            self.top_model_file_name + "_model_best.h5"

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, 
            monitor='loss', 
            save_best_only=True
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
        )

        self.callbacks = [
            # reduce_lr, 
            model_checkpoint,
            early_stopping,
        ]
        
        if self.verbose == True:
            # Show model summary
            self.model.summary()


    def fit(self, trains_set, test_set):
        
        if not tf.test.is_gpu_available:
            print('No GPU was detected. CNNs can be very slow without a GPU.')
        
        
        save_path = None
        model_file_name = None
        
        if self.build_pre_model_flag is True and self.build_top_model_flag is False:
            save_path = self.pre_trained_pre_model_path
            model_file_name = self.pre_model_file_name
        elif self.build_pre_model_flag is True and self.build_top_model_flag is True:
            save_path = self.top_trained_pre_model_path
            model_file_name = self.top_model_file_name
        
        
        if self.build_pre_model_flag is True \
                and self.build_top_model_flag is False \
                and self.source_data_name == "imagenet":
                
            self.model.save_weights(save_path + model_file_name + "_model_last.h5")
            print("Pretrained model saved to: " + save_path)
            return
        
        
        y_train = np.argmax(np.concatenate([y for x, y in trains_set], axis=0), axis=1)
        y_test = np.argmax(np.concatenate([y for x, y in test_set], axis=0), axis=1)
        
        ## Train the pre-model
        # batch_size = 64
        num_epochs = 100
        
        # mini_batch = batch_size
        
        start_time = time.time()
        
        hist = self.model.fit(
            trains_set,
            # batch_size=mini_batch,
            epochs=num_epochs,
            validation_data=test_set,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )
        
        duration = time.time() - start_time
        
        self.model.save(save_path + model_file_name + "_model_last.h5")
        print("Trained transferlearning model saved to: " + save_path)

        y_pred_train = self.predict(trains_set)
        y_pred_test = self.predict(test_set)
        
        # save predictions
        np.save(save_path + model_file_name + '_y_pred_train.npy', y_pred_train)
        np.save(save_path + model_file_name + '_y_pred_test.npy', y_pred_test)

        # convert the predicted from binary to integer
        y_pred_test = np.argmax(y_pred_test, axis=1)
        y_pred_train = np.argmax(y_pred_train, axis=1)
        
        save_logs(save_path+model_file_name, hist, y_train, y_pred_train, y_test, 
            y_pred_test, duration)

        keras.backend.clear_session()
        

    def predict(self, dataset):
        # needs dataset to be a tf.data.Dataset object or any other
        # resulting object like TakeDataset, MapDataset, etc.
        pred = self.model.predict(dataset)
        return pred