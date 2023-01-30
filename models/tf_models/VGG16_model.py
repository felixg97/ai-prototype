
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import time

from keras.models import load_model
from utils.constants import BASE_PATH

"""
1. Use Case: Train a transfer learning model with source domain

(1.1) Build model: Model name, weights, input shape
(1.2) Train model: X_train, y_train
(1.3) Evaluate model: X_test, y_test

2. Use Case: Train the classification layer of the pretrained model with target domain

(2.1) Build pretrained model: Model name, weights, input shape
(2.2) Train classification layer: X_train, y_train
(2.3) Evaluar

"""

class VGG16_model():
    
    def __init__(self, source_data_name, 
                input_shape = None,
                target_data_name = None,
                num_classes = None,
                path = BASE_PATH,
                build_pre_model = False, 
                build_model = False, 
                verbose = False
            ):
        
        self.model_name = "VGG16"
        self.input_shape = input_shape
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.num_classes = num_classes
        self.verbose = verbose
        self.ouput_path = path
        
        self.model = None
        
        self.cross_entropy = None
        if num_classes is not None:
            self.cross_entropy = \
                ("categorical_crossentropy" if num_classes > 2 else "binary_crossentropy")
        
        self.pre_trained_path = path + self.model_name + "_" \
            + self.source_data_name
            
        if build_pre_model == True:
            self.build_pre_model()
            if verbose: self.model.summary()
            return 
        else:
            self.model = load_model(self.pre_trained_path + "_model_init.h5")
        

        ## Ab hier will ich ein ready model haben
        ## Build würde sich jetzt auf das trained layer beziehen
        ## KOMMT SPÄTER: eins nach dem anderen
        # if target_data_name is None:
        #     return
        
        # if build_model == True:
        #     self.build_model()
        #     if verbose: self.mode.summary()
        #     self.model.save_weights(self.output_path + "model_init.hdf5")
        # else: 
        #     self.weights_path = path + "results/models/" + self.model_name + "_" \
        #         + self.source_data_name + "_model_init.h5"
        #     self.model = load_model(self.weights_path)
        
    
    def build_pre_model(self):
        self.model = None # TF model
        
        if self.source_data_name == "imagenet":
            self.model = keras.applications.VGG16(
                input_shape=self.input_shape,
                weights="imagenet",
                include_top=False
            )
            return
        
        self.model = keras.applications.VGG16(
            weights=None,
            input_shape=self.input_shape,
            classes=self.num_classes,
            include_top=True
        )
        
        optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
        
        self.model.compile(
            loss=self.cross_entropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        file_path = self.pre_trained_path + "_model_best.h5"

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, 
            monitor='loss', 
            save_best_only=True
        )

        self.callbacks = [reduce_lr,model_checkpoint]
        
        
        if self.verbose == True:
            print("test")

            # Show model summary
            self.model.summary()
            
    
    def fit_and_save_pre_model(self, 
            X_train, 
            y_train,
            X_test=None, 
            y_test=None
        ):
        
        if not tf.test.is_gpu_available:
            print('No GPU was detected. CNNs can be very slow without a GPU.')
        
        if self.source_data_name == "imagenet":
            self.model.save_weights(self.pre_trained_path + "_last_init.h5")
            print("Pretrained model saved to: " + self.pre_trained_path)
            return
        
        
        ## Train the pre-model
        batch_size = 8
        num_epochs = 100
        
        mini_batch = batch_size
        
        start_time = time.time()
        
        hist = None
        
        if X_test is not None and y_test is not None:
            hist = self.model.fit(
                X_train,
                y_train,
                batch_size=mini_batch,
                epochs=num_epochs,
                validation_data=(X_test, y_test),
                callbacks=self.callbacks,
                verbose=self.verbose,
            )
        else:
            hist = self.model.fit(
                X_train,
                y_train,
                batch_size=mini_batch,
                epochs=num_epochs,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )
        
        duration = time.time() - start_time
        
        self.model.save(self.pre_trained_path + "_model_last.h5")
        print("Trained transferlearning model saved to: " + self.pre_trained_path)
        
        # if X_test is not None and y_test is not None:
        #     y_pred = self.model.predict(X_test, y_test)
        #     # save predictions
        #     np.save(self.pre_trained_path + "_ypred.npy", y_pred)
        
        keras.backend.clear_session()
            
            
    def build_model(self):

        if self.source_data_name == "imagenet":
            pass
        
        # do if not imagenet
        
        if self.source_data_name != "imagenet":
            # freeze all layers except last three (or four?)
            # deactivate last three layers and add custom layers
            pass
        
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
            self.num_classes, activation="softmax", name="predictions")(x)
        
        self.model = keras.models.Model(
            input=+self.model.input, 
            outputs=output_layer)
        
        # Usa RMSprop optimizer
        optimizer = keras.optimizers.RMSprop(learning_rate=1e-5)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer, 
            loss=self.cross_entropy, 
            metrics=['accuracy']
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        file_path = self.pre_trained_path + "_model_best.h5"

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=file_path, 
            monitor='loss', 
            save_best_only=True
        )

        self.callbacks = [reduce_lr,model_checkpoint]
        
        if self.verbose == True:
            # Show model summary
            self.model.summary()

        
    def fit_model(self, X_train, y_train, X_test=None, y_test=None):
        if not tf.test.is_gpu_available:
            print('No GPU was detected. CNNs can be very slow without a GPU.')
            
        num_epochs = 50
        self.channels = 1
        
        start_time = time.time()
        
        hist = None
        
        if X_test is not None and y_test is not None:
            hist = self.model.fit(
                X_train,
                y_train,
                epochs=num_epochs,
                validation_data=(X_test, y_test),
                callbacks=self.callbacks,
                verbose=self.verbose,
            )
        else:
            hist = self.model.fit(
                X_train,
                y_train,
                epochs=num_epochs,
                callbacks=self.callbacks,
                verbose=self.verbose,
            )
            
        duration = time.time() - start_time
        
        self.model.save(self.pre_trained_path + "_model_last.h5")
        print("Trained transferlearning model saved to: " + self.pre_trained_path)
        
        if X_test is not None and y_test is not None:
            y_pred = self.model.predict(X_test, y_test)
            # save predictions
            np.save(self.pre_trained_path + "_ypred.npy", y_pred)
        
        keras.backend.clear_session()
        

    def predict(self):
        pass

        
        
    
    
    