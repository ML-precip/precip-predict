from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv2D, BatchNormalization,Flatten, MaxPooling2D
from keras.layers import Conv2DTranspose,Concatenate,UpSampling2D,Cropping2D
from keras.layers import Input, Lambda, Reshape, Dropout, Activation

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def crop_output(u1,c1):
        h, w = c1.shape[1:3]  # reconstructed width and hight
        h_tgt, w_tgt = u1.shape[1:3]
        dh = h - h_tgt  # deltas to be cropped away
        dw = w - w_tgt
        crop = Cropping2D(cropping=((dh//2, dh-dh//2), (dw//2, dw-dw//2)))(c1)
        return(crop)
    
    
    
    
    
def Unet1(input_s, num_filters, ks, activation):
    """General U-Net architecture
    it consits on 2 main parts: enconder-decoder
    In this model UpSampling is used to upscale
    no dropout or bachtnormalization"""
    
    img = Input(shape=input_s)
    x = img
    pad = 'same'
    
    # Encoder part
    conv1 = Conv2D(num_filters,kernel_size=ks,padding=pad,activation=activation)(x)
    conv1 = Conv2D(num_filters,kernel_size=ks,padding=pad,activation=activation)(conv1)#
    max_pool1 = MaxPool2D((2,2))(conv1)
    print('Max pool 1:',max_pool1.shape)

    conv2 = Conv2D(num_filters*2,kernel_size=ks,padding=pad,activation=activation)(max_pool1)
    conv2 = Conv2D(num_filters*2,kernel_size=ks,padding=pad,activation=activation)(conv2)#
    max_pool2 = MaxPool2D((2,2))(conv2)
    print('Max pool 2:',max_pool2.shape)

    conv3 = Conv2D(num_filters*4,kernel_size=ks,padding=pad,activation=activation)(max_pool2)
    conv3 = Conv2D(num_filters*4,kernel_size=ks,padding=pad,activation=activation)(conv3) 
    max_pool3 = MaxPool2D((2,2))(conv3)
    print('Max pool 3:',max_pool3.shape)

    conv4 = Conv2D(num_filters*8,kernel_size=ks,padding=pad,activation=activation)(max_pool3)
    conv4 = Conv2D(num_filters*8,kernel_size=ks,padding=pad,activation=activation)(conv4)  #
    max_pool4 = MaxPool2D((2,2))(conv4)
    print('Max pool 4:',max_pool4.shape)
    print('conv 4',conv4.shape)

    conv5 = Conv2D(num_filters*16,kernel_size=ks,padding=pad,activation=activation)(max_pool4)
    conv5 = Conv2D(num_filters*16,kernel_size=ks, padding=pad,activation=activation)(conv5)
    print('Down sampling last layers output shape:',conv5.shape)
    
    
    # deconding part
    # Upsampling 
    # upconv1 
    up_conv1 = UpSampling2D((2,2))(conv5)
    print(up_conv1.shape) 
    crop= crop_output(up_conv1, conv4)
    merged = Concatenate()([crop, up_conv1])
    up_conv_conv1 = Conv2D(num_filters*8,kernel_size=ks, padding=pad,activation=activation)(merged)
    up_conv_conv1= Conv2D(num_filters*8,kernel_size=ks,padding=pad,activation=activation)(up_conv_conv1)
    print('Up-conv 1:',up_conv_conv1.shape)
    
    # upconv2 
    up_conv2 = UpSampling2D((2,2))(up_conv_conv1)
    print(up_conv2)
    crop2 = crop_output(up_conv2, conv3)
    print(crop2.shape)
    merged = Concatenate()([up_conv2, crop2])
    #         print(merged.shape)
    up_conv_conv2 = Conv2D(num_filters*4,kernel_size=ks, padding=pad,activation=activation)(merged)
    up_conv_conv2 = Conv2D(num_filters*4,kernel_size=ks, padding=pad,activation=activation)(up_conv_conv2)
    print('Up_conv 2:',up_conv_conv2.shape)
    
     # upconv3
    up_conv3 = UpSampling2D((2,2))(up_conv_conv2)
    crop3 = crop_output(up_conv3, conv2)      
    merged = Concatenate()([up_conv3, crop3])
    print('Conv 3 merger',merged.shape)
    up_conv_conv3 = Conv2D(num_filters*2,kernel_size=ks, padding=pad,activation=activation)(merged)
    up_conv_conv3 = Conv2D(num_filters*2,kernel_size=ks, padding=pad,activation=activation)(up_conv_conv3)
    print('Up-conv 3:',up_conv_conv3.shape)
    
    # upconv4
    up_conv4 = UpSampling2D((2,2))(up_conv_conv3)
    crop4 = crop_output(up_conv4, conv1)      
    merged = Concatenate()([up_conv4, crop4])
    up_conv_conv4 = Conv2D(num_filters*1,kernel_size=ks,padding=pad,activation=activation)(merged)
    up_conv_conv4 = Conv2D(num_filters*1,kernel_size=ks, padding=pad,activation=activation)(up_conv_conv4)
    print('Up-conv 4:',up_conv_conv4.shape)

    out = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')(up_conv_conv4)
    model = Model(inputs=img,outputs=out)
    
    return(model)





# Encoder part
def build_encoder_block(previous_layer, filters, activation, use_batchnorm, dropout):
    """Encoder part for the class Unet2"""
    c = Conv2D(filters, (3, 3), activation=activation,
             kernel_initializer='he_normal', padding='same')(previous_layer)
    if use_batchnorm:
        c = BatchNormalization()(c)
    if dropout:
        c = Dropout(0.2)(c)
    c = Conv2D(filters, (3, 3), activation=activation,
                kernel_initializer='he_normal', padding='same')(c)
    if use_batchnorm:
        c = BatchNormalization()(c)
    p = MaxPooling2D((2, 2))(c)

    return c, p




# decoder part
def build_decoder_block(previous_layer, skip_layer, is_last, filters, activation, use_batchnorm, dropout):
    """Decoder part for the class Unet2"""
    u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                        padding='same')(previous_layer)
    u = Concatenate()([u, skip_layer])
    c = Conv2D(filters, (3, 3), activation=activation, 
               kernel_initializer='he_normal', padding='same')(u)
    if use_batchnorm:
        c = BatchNormalization()(c)
    if dropout:
        c = Dropout(0.2)(c)
    c = Conv2D(filters, (3, 3), activation=activation, 
               kernel_initializer='he_normal', padding='same')(c)
    if use_batchnorm and not is_last:
        c = BatchNormalization()(c)

    return c


def build_bottleneck(previous_layer, filters, activation, use_batchnorm, dropout):
    """Bottelenck to join the Encod-Decod and complete the U-net"""
    c = Conv2D(filters, (3, 3), activation=activation, 
               kernel_initializer='he_normal', padding='same')(previous_layer)
    if use_batchnorm:
        c = BatchNormalization()(c)
    if dropout:
        c = Dropout(0.2)(c)
    c = Conv2D(filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(c)
    if use_batchnorm:
        c = BatchNormalization()(c)

    return c



class Unet2():
    """Similar to Unet1 but using batchnorm- dropout
       the decoder part uses Conv2DTranspose"""
    def __init__(self, input_s, output_channels ):
        self.input_s = input_s
        self.output_channels = output_channels
        
    def build_model(self):
    
        inputs = Input(self.input_s)
            #inputs = Input(shape=self.input_s)
        x1, pp1 = build_encoder_block(inputs, 64,  "relu", True, True)
        x2, pp2 = build_encoder_block(pp1, 128,  "relu", True, True)
        x3, pp3 = build_encoder_block(pp2, 256,  "relu", True, True)
        x4, pp4 = build_encoder_block(pp3, 512,  "relu", True, True)

        tb = build_bottleneck(pp4, 1024, "relu", True, True)

        dd1 = build_decoder_block(tb, x4, False, 512, "relu", True, True)
        dd2 = build_decoder_block(dd1, x3,False, 256, "relu", True, True)
        dd3 = build_decoder_block(dd2, x2,False, 128, "relu", True, True)
        dd4 = build_decoder_block(dd3, x1, True, 64,"relu", True, True)

        output_function = 'softmax' if self.output_channels > 1 else 'sigmoid'
        ouput_layer     = Conv2D(self.output_channels, (1, 1),
                                         activation=output_function)(dd4)
        u_model = Model(inputs=inputs,outputs=ouput_layer)

        return(u_model)