import pandas as pd
import numpy as np

class CVAEParams(object):
    '''
    classdocs
    '''

    def __init__(self, image_shape,
                 num_classes,
                 type_layers="dense",
                 num_encoding_layers=3,
                 num_decoding_layers=3,
                 kernel_size=3,
                 stride=2,
                 intermediate_dim=512,
                 var_x=1.0,
                 r_loss="mse",
                 latent_dim = 3,
                 learning_rate = 0.001,
                 double_convolution = False,
                 steps = 10,
                 constant_t = False,
                 constant_logt_value = 0.01):
        '''
        Constructor
        '''

        # Data parameters
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.image_size = np.product(image_shape)
        self.latent_dim = latent_dim
        # Architecture parameters
        self.type_layers = type_layers
        self.intermediate_dim = intermediate_dim
        self.num_encoding_layers = num_encoding_layers
        self.num_decoding_layers = num_decoding_layers
        if self.type_layers == "convolutional":
            self.kernel_size = int(kernel_size)
            self.stride = int(stride)
        else:
            self.kernel_size = 0
            self.stride = 0
        self.double_convolution = double_convolution

        self.r_loss = r_loss
        self.learning_rate = learning_rate


        # Diffusion parameters
        self.steps = steps
        self.constant_t = constant_t
        self.constant_logt_value = constant_logt_value

        # Decoder parameters
        self.var_x = var_x



    def params_to_df(self):
        data = {"image_shape": [self.image_shape],
                "num_classes":[self.num_classes],
                "latent_dim" : [self.latent_dim],
                "var_x":[self.var_x],
                "intermediate_dim": [self.intermediate_dim],
                "r_loss": [self.r_loss],
                "type_layers": [self.type_layers],
                "num_encoding_layers": [self.num_encoding_layers],
                "num_decoding_layers": [self.num_decoding_layers],
                "kernel_size": [self.kernel_size],
                "stride": [self.stride],
                "double_convoltion": [self.double_convolution]}
        df = pd.DataFrame(data)
        return df
