import keras
from keras import initializers 
from keras import regularizers 
from keras import constraints 
from keras import layers
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv2D, BatchNormalization,Flatten, MaxPooling2D
from keras.layers import Conv2DTranspose,Concatenate,UpSampling2D,Cropping2D, SpatialDropout2D
from keras.layers import Input, Lambda, Reshape, Dropout, Activation, ZeroPadding2D, Concatenate
from keras.layers import Activation, Reshape, Flatten, ConvLSTM2D
import numpy as np
import pandas as pd
import geopandas as gpd

class DeepFactory_Keras():
    """
    Model factory.
    """

    def __init__(self, arch, input_size, output_size, for_extremes=False, latent_dim=128, 
                 dropout_rate=0.2, use_batch_norm=True, inner_activation='relu', unet_filters_nb=64, 
                 unet_depth=4, unet_use_upsample=True, output_scaling=1, output_crop=None):
        super(DeepFactory_Keras, self).__init__()
        self.arch = arch
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        self.for_extremes = for_extremes
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.unet_use_upsample = unet_use_upsample
        self.inner_activation = inner_activation
        self.unet_depth = unet_depth
        self.unet_filters_nb = unet_filters_nb
        self.output_scaling = output_scaling
        self.output_crop = output_crop
        
        self.last_activation = 'relu'
        if for_extremes:
            self.last_activation = 'sigmoid'

        if arch == 'Davenport-2021':
            self.build_Davenport_2021()
        elif arch == 'CNN-2L':
            self.build_CNN_2L()
        elif arch == 'Unet':
            self.build_Unet()
        elif arch == 'Unet-attention':
            self.build_UnetAtt()
        elif arch == 'Pan-2019':
            self.build_Pan_2019()
        elif arch =='Conv-LTSM':
            self.build_convLTSM()
        else:
            raise ValueError('The architecture was not correctly defined')
        
        
    def build_Davenport_2021(self):
        """
        Based on: Davenport, F. V., & Diffenbaugh, N. S. (2021). Using Machine Learning to 
        Analyze Physical Causes of Climate Change: A Case Study of U.S. Midwest Extreme Precipitation. 
        Geophysical Research Letters, 48(15). https://doi.org/10.1029/2021GL093787
        """
        
        # Downsampling
        inputs = Input(shape=self.input_size)
        x = Conv2D(16, 3, padding='same', activity_regularizer=regularizers.l2(0.01))(inputs)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = SpatialDropout2D(self.dropout_rate)(x) # In original: simple Dropout
        x = Conv2D(16, 3, padding='same', activity_regularizer=regularizers.l2(0.01))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = SpatialDropout2D(self.dropout_rate)(x) # In original: simple Dropout
        x = Flatten()(x)                
        x = Dense(self.latent_dim, activity_regularizer=regularizers.l2(0.001))(x) # In original: 16
        x = Activation('relu')(x)

        next_shape = self.get_shape_for(stride_factor=4)

        # Upsampling. In original: no decoder
        x = self.dense_block(x, np.prod(next_shape))
        x = Reshape(target_shape=next_shape)(x)
        x = self.deconv_block(x, 16, 3, stride=2)
        x = self.deconv_block(x, 16, 3, stride=2)
        x = self.conv_block(x, 1, 3, activation=self.last_activation)
        outputs = self.final_cropping_block(x)
 
        self.model = keras.Model(inputs, outputs, name="Davenport-2021")
        
        
    def build_Pan_2019(self):
        """
        Based on: Pan, B., Hsu, K., AghaKouchak, A., & Sorooshian, S. (2019). 
        Improving Precipitation Estimation Using Convolutional Neural Network. 
        Water Resources Research, 55(3), 2301–2321. https://doi.org/10.1029/2018WR024090
        """
        # In original: padding='valid'
        
        # Downsampling
        inputs = Input(shape=self.input_size)
        x = Conv2D(15, 4, padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(20, 4, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(20, 4, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.latent_dim, activation='relu')(x) # In original: 60

        next_shape = self.get_shape_for(stride_factor=4)

        # Upsampling. In original: no decoder
        x = self.dense_block(x, np.prod(next_shape))
        x = Reshape(target_shape=next_shape)(x)
        x = self.deconv_block(x, 20, 4, stride=2)
        x = self.deconv_block(x, 20, 4, stride=2)
        x = self.conv_block(x, 15, 4)
        x = self.conv_block(x, 1, 3, activation=self.last_activation)
        outputs = self.final_cropping_block(x)

        self.model = keras.Model(inputs, outputs, name="Pan-2019")


    def build_CNN_2L(self):

        # Downsampling
        inputs =Input(shape=self.input_size)
        x = self.conv_block(inputs, 32, 3, stride=2, with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, 64, 3, stride=2, with_batchnorm=True, with_dropout=True)
        x = Flatten()(x)
        x = self.dense_block(x, self.latent_dim, activation='sigmoid')

        next_shape = self.get_shape_for(stride_factor=4)

        # Upsampling
        x = self.dense_block(x, np.prod(next_shape))
        x = Reshape(target_shape=next_shape)(x)
        x = self.deconv_block(x, 64, 3, stride=2)
        x = self.deconv_block(x, 32, 3, stride=2)
        x = self.deconv_block(x, 1, 3, activation=self.last_activation)
        outputs = self.final_cropping_block(x)

        self.model = keras.Model(inputs, outputs, name="CNN-v1")
        

    def build_convLTSM(self):
        
        expanded_shape = self.input_size.copy()
        expanded_shape.insert(0, 1)
        
        inputs = Input(shape=self.input_size)
        
        x = Reshape(target_shape=expanded_shape)(inputs)
        
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=True, padding = 'same',
                              go_backwards=True, activation='tanh', data_format = 'channels_last',
                              dropout=0.4, recurrent_dropout=0.2)(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=16, kernel_size=(3, 3), return_sequences=True, padding = 'same',
                              go_backwards=True, activation='tanh', data_format = 'channels_last',
                              dropout=0.4, recurrent_dropout=0.2)(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=8, kernel_size=(3, 3), return_sequences=False, padding = 'same',
                              go_backwards=True, activation='tanh', data_format = 'channels_last',
                              dropout=0.3, recurrent_dropout=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', data_format='channels_last')(x)

        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        #model.add(Flatten())
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(2, 2),  activation='sigmoid',
                                   padding='same', data_format='channels_last')(x)

        outputs = self.final_cropping_block(x)

        self.model = keras.Model(inputs, outputs, name="Conv-LTSM")
        
        
    def build_Unet(self):
        """
        Based on: U-Net: https://github.com/nikhilroxtomar/Unet-for-Person-Segmentation/blob/main/model.py
        """
        
        # Downsampling
        inputs = Input(shape=self.input_size)
        
        # Pad if necessary
        x = self.padding_block(inputs, factor=2**self.unet_depth)
        
        skips = []
        for i in range(self.unet_depth):
            s, x = self.unet_encoder_block(x, self.unet_filters_nb * 2**i)
            skips.append(s)
        
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True)

        # Upsampling
        for i in reversed(range(self.unet_depth)):
            x = self.unet_decoder_block(x, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
        
        # Additional upsampling for downscaling
        x = self.handle_output_scaling(x)

        x = self.conv_block(x, 1, 1, activation=self.last_activation)
        outputs = self.final_cropping_block(x)

        self.model = keras.Model(inputs, outputs, name="U-Net-v1")
                
            
    def build_UnetAtt(self):
        """
        U-Net with attention mechanism
        """
        
        # Downsampling
        inputs = Input(shape=self.input_size)
        
        # Pad if necessary
        x = self.padding_block(inputs, factor=self.unet_depth**2)
        
        skips = []
        for i in range(self.unet_depth):
            s, x = self.unet_encoder_block(x, self.unet_filters_nb * 2**i)
            skips.append(s)
        
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, self.unet_filters_nb * 2**self.unet_depth, 3, initializer='he_normal', with_batchnorm=True)

        # Upsampling
        for i in reversed(range(self.unet_depth)):
            x = self.unet_decoder_block_attention(x, skips[i], self.unet_filters_nb * 2**i, is_last=(i==0))
        
        # Additional upsampling for downscaling
        x = self.handle_output_scaling(x)

        x = self.conv_block(x, 1, 1, activation=self.last_activation)
        outputs = self.final_cropping_block(x)

        self.model = keras.Model(inputs, outputs, name="U-Net-attention")
        
        
    def unet_encoder_block(self, input, filters, kernel_size=3):
        x = self.conv_block(input, filters, kernel_size, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, filters, kernel_size, initializer='he_normal', with_batchnorm=True)
        p = MaxPooling2D((2, 2))(x)
        
        return x, p

    
    def unet_decoder_block(self, input, skip_features, filters, conv_kernel_size=3, deconv_kernel_size=2, is_last=False):
        x = self.deconv_block(input, filters, deconv_kernel_size, stride=2)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=(not is_last))

        return x
        
        
    def unet_decoder_block_attention(self, input, skip_features, filters, conv_kernel_size=3, deconv_kernel_size=2, is_last=False):
        x = attention_up_and_concate(input, skip_features, data_format='channels_last')
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=True, with_dropout=True)
        x = self.conv_block(x, filters, conv_kernel_size, initializer='he_normal', with_batchnorm=(not is_last))

        return x
        
        
    def conv_block(self, input, filters, kernel_size=3, stride=1, padding='same', initializer='default', activation='default', 
                   with_batchnorm=False, with_pooling=False, with_dropout=False, with_late_activation=False):
        if activation == 'default':
            activation = self.inner_activation
            
        conv_activation = activation
        if with_late_activation:
            conv_activation = None
            
        if initializer == 'default':
            x = Conv2D(filters, kernel_size, strides=(stride, stride), padding=padding, activation=conv_activation)(input)
        else:
            x = Conv2D(filters, kernel_size, strides=(stride, stride), padding=padding, activation=conv_activation, kernel_initializer=initializer)(input)
            
        if with_batchnorm:
            x = BatchNormalization()(x)
        if with_late_activation:
            x = Activation(activation)(x)
        if with_pooling:
            x = MaxPooling2D(pool_size=2)(x)
        if with_dropout:
            x = SpatialDropout2D(self.dropout_rate)(x)
        
        return x
    
    
    def deconv_block(self, input, filters, kernel_size=3, stride=1, padding='same', initializer='default', activation='default', 
                     with_batchnorm=False, with_dropout=False):
        if activation == 'default':
            activation = self.inner_activation
        
        if self.unet_use_upsample:
            x = UpSampling2D((2, 2))(input)
        else:
            if initializer == 'default':
                x = Conv2DTranspose(filters, kernel_size, strides=stride, padding=padding, activation=activation)(input)
            else:
                x = Conv2DTranspose(filters, kernel_size, strides=stride, padding=padding, activation=activation, kernel_initializer=initializer)(input)
            
        if with_batchnorm:
            x = BatchNormalization()(x)
        if with_dropout:
            x = SpatialDropout2D(self.dropout_rate)(x)
        
        return x
    
    
    def dense_block(self, input, units, activation='default', with_dropout=False):
        if activation == 'default':
            activation=self.inner_activation
            
        x = Dense(units, activation=activation)(input)
        if with_dropout:
            x = Dropout(self.dropout_rate)(x)
            
        return x

    
    def handle_output_scaling(self, input, with_batchnorm=False):
        if self.output_scaling > 1:
            if self.output_scaling == 2:
                x = self.deconv_block(input, 64, 3, stride=2, with_batchnorm=with_batchnorm)
            elif self.output_scaling == 3:
                x = self.deconv_block(input, 64, 3, stride=3, with_batchnorm=with_batchnorm)
            elif self.output_scaling == 4:
                x = self.deconv_block(input, 64, 3, stride=2, with_batchnorm=with_batchnorm)
                x = self.deconv_block(x, 64, 3, stride=2, with_batchnorm=with_batchnorm)
            elif self.output_scaling == 5:
                x = self.deconv_block(input, 64, 3, stride=3, with_batchnorm=with_batchnorm)
                x = self.deconv_block(x, 64, 3, stride=2, with_batchnorm=with_batchnorm)
            else:
                raise NotImplementedError('Level of downscaling not implemented')
        else:
            x = input
        
        if self.output_crop:
            raise NotImplementedError('Manual cropping not yet implemented')
            
        return x
            
        
    def padding_block(self, x, factor):
        h, w = x.get_shape().as_list()[1:3]
        dh = 0
        dw = 0
        if h % factor > 0:
            dh = factor - h % factor
        if w % factor > 0:
            dw = factor - w % factor
        if dh > 0 or dw > 0:
            top_pad = dh//2
            bottom_pad = dh//2 + dh%2
            left_pad = dw//2
            right_pad = dw//2 + dw%2
            x = ZeroPadding2D(padding=((top_pad, bottom_pad), (left_pad, right_pad)))(x)
        
        return x
        
        
    def final_cropping_block(self, x):
        # Compute difference between reconstructed width and hight and the desired output size.
        h, w = x.get_shape().as_list()[1:3]
        h_tgt, w_tgt = self.output_size[:2]
        dh = h - h_tgt
        dw = w - w_tgt

        if dh < 0 or dw < 0:
            raise ValueError(f'Negative values in output cropping dh={dh} and dw={dw}')

        # Add to decoder cropping layer and final reshaping
        x = Cropping2D(cropping=((dh//2, dh-dh//2), (dw//2, dw-dw//2)))(x)
        #x = layers.Reshape(target_shape=self.output_size,)(x)
        
        return x
        

    def get_shape_for(self, stride_factor):
        next_shape = self.output_size.copy()
        next_shape[0] = int(np.ceil(next_shape[0]/stride_factor))
        next_shape[1] = int(np.ceil(next_shape[1]/stride_factor))

        return next_shape

        
    def call(self, x):
        return self.model(x)
    
