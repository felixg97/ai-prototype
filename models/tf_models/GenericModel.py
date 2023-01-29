
import tensorflow as tf


class GenericModel():
    
    def __init__(self, weights="imagenet", 
                include_top=False, 
                input_shape=(350, 350, 3),
                dataset_name="local",
                checkpoint_path="/Users/felixgerschenr/git/ai-prototype/results/", 
                verbose=False):
        
        self.model_name = "VGG16"

        
        self.build_model()
    
    def build_model(self):
        self.model = None # TF model
        
        #

        #
        
        if self.verbose == True:
            print("test")
            
            # Show model summary
            self.model.summary()

        
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        if not tf.test.is_gpu_available:
            print('No GPU was detected. CNNs can be very slow without a GPU.')
        



    def predict(self):
        pass
    
    
    