from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model
#Conv-BatchNorm-ReLU block
def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
#Identity block
def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = Add()([tensor,x])    #skip connection
    x = ReLU()(x)
    return x

#Projection block 
 
def projection_block(tensor, filters, strides): 
    #left stream     
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)     
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)     
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)     
    x = BatchNormalization()(x) 
         
    #right stream     
    shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)     
    shortcut = BatchNormalization()(shortcut)          
    x = Add()([shortcut,x])    #skip connection     
    x = ReLU()(x)          
    return x


def resnet_block(x, filters, reps, strides):
    
    x = projection_block(x, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
    return x

