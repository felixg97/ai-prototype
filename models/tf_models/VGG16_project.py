import os
import time
import tensorflow as tf

from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras import backend as K

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import get_custom_objects

class L2SP(Regularizer):
    
    def __init__(self, alpha=0., beta=0., x0=[]):
        if isinstance(x0, dict):
            x0 = x0['value']
        
        self.alpha = K.cast_to_floatx(alpha)
        self.beta = K.cast_to_floatx(beta)
        self.x0_tensor = K.constant(K.cast_to_floatx(x0))
        self.x0 = x0
    
    def __call__(self, x):
        regularization = 0.
        
        if self.alpha:
            regularization += K.sum(self.alpha / 2 * K.square(x - self.x0_tensor))
        if self.beta:
            regularization += K.sum(self.beta / 2 * K.square(x))
        
        return regularization
    
    def get_config(self):
        return {'alpha': float(self.alpha),
                'beta': float(self.beta),
                'x0': self.x0}

    def l2sp(alpha=0., beta=0., x0=[]):
        return L2SP(alpha, beta, x0)


class VGG16_model:
    
    def __init__(self, weights="imagenet", 
                include_top=False, 
                input_shape=(350, 350, 3), 
                alpha=.1, 
                beta=.01,
                dataset_name="local",
                checkpoint_path="/Users/felixgerschenr/git/ai-prototype/results/", 
                verbose=False):
        
        self.model_name = "VGG16"
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path += dataset_name + "/" + self.model_name + ".hdf5"
        
        self.weights = weights
        self.include_top = include_top
        self.input_shape = input_shape
        
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        
        self.build_model()
    
    def build_model(self):
        # Create pretrained Xception model using Imagenet weights, 
        # but without the original classifier
        self.model = VGG16(weights=self.weights, 
                        include_top=self.include_top, 
                        input_shape=self.input_shape)
        
        # Add L2-SP regularizer to custom objects 
        get_custom_objects().update({"L2SP": L2SP.l2sp})
        
        # Add L2-SP regularizer to pretrained layers
        for index in range(len(self.model.layers)):
            if isinstance(self.model.layers[index], tf.keras.layers.Conv2D):
                self.model.layers[index].kernel_regularizer = \
                    L2SP.l2sp(alpha=self.alpha, beta=self.beta, 
                            x0=self.model.layers[index].get_weights()[0])
                    
        
        # Compile to suppress warning
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        
        
        # Save and load to make new loss work
        self.model.save('temp_vgg.hdf5')
        self.model = load_model('temp_vgg.hdf5')
        os.remove('temp_vgg.hdf5')
        
        # Add new classifier / last layer
        x = self.model.layers[-1].output
        x = layers.Flatten()(x)
        x = layers.Dense(32,
                activation=None,
                kernel_regularizer=L2SP.l2sp(beta=self.beta),
                use_bias=False,
                name='fc1')(x)
        x = layers.Dropout(0.8, name='dropout_1')(x)
        x = layers.BatchNormalization(name='fc1_bn')(x)
        x = layers.Activation('relu', name='fc1_act')(x)
        output = layers.Dense(2, activation='sigmoid', name='predictions')(x)

        self.model = Model(self.model.input, output)
        
        # Usa RMSprop optimizer
        optimizer = RMSprop(learning_rate=1e-5)
        
        # Compile model
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                    metrics=['accuracy'])
        
        if self.verbose == True:
            print("test")
            
            # Show model summary
            self.model.summary()

        
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        if not tf.test.is_gpu_available:
            print('No GPU was detected. CNNs can be very slow without a GPU.')
        
        # Number of training epochs
        self.nb_epoch = 50

        # Color mode grayscale
        self.channels = 1
        
        
        model_checkpoint = ModelCheckpoint(self.checkpoint_path, 
                                        save_best_only = True)
        
        self.callbacks = []
        
        # Track train duration
        start_time = time.time() 

        hist = self.model.fit(
            X_train, 
            y_train,
            validation_data = (X_test, y_test),
            steps_per_epoch=25, 
            epochs=self.nb_epoch,
            verbose=self.verbose, 
            callbacks=self.callbacks)
        
        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = tf.models.load_model(self.output_directory+'best_model.hdf5')

        y_pred = model.predict()

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()
        



    def predict(self):
        pass
    
    
    