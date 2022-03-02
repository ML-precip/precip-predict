from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv2D, BatchNormalization,Flatten, MaxPooling2D
from keras.layers import Conv2DTranspose,Concatenate,UpSampling2D,Cropping2D
from keras.layers import Input, Lambda, Reshape, Dropout, Activation, ZeroPadding2D

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def crop_output(u1,c1):
    # u1: layer with the wanted shapes
    # c1: layer to crop out
    h, w = c1.shape[1:3]  # reconstructed width and hight
    h_tgt, w_tgt = u1.shape[1:3]
    dh = h - h_tgt  # deltas to be cropped away
    dw = w - w_tgt
        
    if dh < 0 or dw < 0:
        raise('Negative values in output cropping')
            
    crop = Cropping2D(cropping=((dh//2, dh-dh//2), (dw//2, dw-dw//2)))(c1)
    return(crop)
    
    
    
    
    
def Unet1(input_s, num_filters, ks, activation):
    """General U-Net architecture
    it consits on 2 main parts: enconder-decoder
    In this model UpSampling is used to upscale
    no dropout or bachtnormalization"""
    # Adapted from: https://github.com/work-mohit/U-Net_From-Scratch/blob/master/UNet.ipynb
    
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
    # Upsampling-> change to Conv2D 
    # upconv1 
    #up_conv1 = UpSampling2D((2,2))(conv5)
    up_conv1 = Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(conv5)
    print(up_conv1.shape) 
    crop= crop_output(up_conv1, conv4)
    merged = Concatenate()([crop, up_conv1])
    up_conv_conv1 = Conv2D(num_filters*8,kernel_size=ks, padding=pad,activation=activation)(merged)
    up_conv_conv1= Conv2D(num_filters*8,kernel_size=ks,padding=pad,activation=activation)(up_conv_conv1)
    print('Up-conv 1:',up_conv_conv1.shape)
    
    # upconv2 
    #up_conv2 = UpSampling2D((2,2))(up_conv_conv1)
    up_conv2 = Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(up_conv_conv1)
    print(up_conv2)
    crop2 = crop_output(up_conv2, conv3)
    print(crop2.shape)
    merged = Concatenate()([up_conv2, crop2])
    #         print(merged.shape)
    up_conv_conv2 = Conv2D(num_filters*4,kernel_size=ks, padding=pad,activation=activation)(merged)
    up_conv_conv2 = Conv2D(num_filters*4,kernel_size=ks, padding=pad,activation=activation)(up_conv_conv2)
    print('Up_conv 2:',up_conv_conv2.shape)
    
     # upconv3
    #up_conv3 = UpSampling2D((2,2))(up_conv_conv2)
    up_conv3 = Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(up_conv2)
    crop3 = crop_output(up_conv3, conv2)      
    merged = Concatenate()([up_conv3, crop3])
    print('Conv 3 merger',merged.shape)
    up_conv_conv3 = Conv2D(num_filters*2,kernel_size=ks, padding=pad,activation=activation)(merged)
    up_conv_conv3 = Conv2D(num_filters*2,kernel_size=ks, padding=pad,activation=activation)(up_conv_conv3)
    print('Up-conv 3:',up_conv_conv3.shape)
    
    # upconv4
    
    #up_conv4 = UpSampling2D((2,2))(up_conv_conv3)
    # I need to increase the strides to match the size!
    up_conv4 = Conv2DTranspose(num_filters*1, (2, 2), strides=(2, 2), padding='same')(up_conv3)
    # and crop accordingly
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
   
    skip_layer= crop_output(u, skip_layer)    
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

# Adding padding and cropping parts -- 
            
        
def padding_block(x, factor):
    
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
        
        
def final_cropping_block(x):
    # Compute difference between reconstructed width and hight and the desired output size.
    h, w = x.get_shape().as_list()[1:3]
    h_tgt, w_tgt = self.output_size[:2]
    dh = h - h_tgt
    dw = w - w_tgt

    if dh < 0 or dw < 0:
        raise('Negative values in output cropping')

    # Add to decoder cropping layer and final reshaping
    x = Cropping2D(cropping=((dh//2, dh-dh//2), (dw//2, dw-dw//2)))(x)
    x = Reshape(target_shape=self.output_size,)(x)
        
    return x
        

def get_size_for(stride_factor):
    next_shape = output_size.copy()
    next_shape[0] = int(np.ceil(next_shape[0]/stride_factor))
    next_shape[1] = int(np.ceil(next_shape[1]/stride_factor))

    return next_shape



class Unet2():
    """Similar to Unet1 but using batchnorm- dropout
       the decoder part uses Conv2DTranspose"""
    # Adapted from https://github.com/nikhilroxtomar/Unet-for-Person-Segmentation/blob/main/model.py
    def __init__(self, input_s, output_size, output_channels, num_filters, use_batchnorm, dropout):
        self.input_s = input_s
        self.output_channels = output_channels
        self.output_size = output_size
        self.num_filters = num_filters
        self.use_batchnorm = use_batchnorm
        self.droput = dropout
        
    def build_model(self):
    
        inputs = Input(self.input_s)
        # Additional padding if needed 
        x = padding_block(inputs, factor=16)
            
        x1, pp1 = build_encoder_block(x, self.num_filters*2,  "relu", self.use_batchnorm, self.droput)
        x2, pp2 = build_encoder_block(pp1, self.num_filters*4,  "relu", self.use_batchnorm, self.droput)
        x3, pp3 = build_encoder_block(pp2, self.num_filters*8,  "relu", self.use_batchnorm, self.droput)
        x4, pp4 = build_encoder_block(pp3, self.num_filters*16,  "relu", self.use_batchnorm, self.droput)

        tb = build_bottleneck(pp4, self.num_filters*32, "relu", True, True)

        dd1 = build_decoder_block(tb, x4, False, self.num_filters*16, "relu", self.use_batchnorm, self.droput)
        dd2 = build_decoder_block(dd1, x3,False, self.num_filters*8, "relu", self.use_batchnorm, self.droput)
        dd3 = build_decoder_block(dd2, x2,False, self.num_filters*4, "relu", self.use_batchnorm, self.droput)
        dd4 = build_decoder_block(dd3, x1, True, self.num_filters*2,"relu", self.use_batchnorm, self.droput)

        output_function = 'softmax' if self.output_channels > 1 else 'sigmoid'
        output_layer     = Conv2D(self.output_channels, (1, 1),
                                         activation=output_function)(dd4)
        # Additional cropping
        
        output_layer = crop_output(inputs,output_layer)
        u_model = Model(inputs=inputs,outputs=output_layer)

        return(u_model)