from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Conv2DTranspose
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model
import tensorflow as tf

#https://towardsdatascience.com/creating-deeper-bottleneck-resnet-from-scratch-using-tensorflow-93e11ff7eb02

def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    
    """Conv-BatchNorm-ReLU block"""
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def identity_block(inputs, filters):
    """Identity block"""
    x = conv_batchnorm_relu(inputs, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = Add()([inputs,x])    #skip connection
    x = ReLU()(x)
    return x


 
def projection_block(inputs, filters, strides=2): 
    """Projection block """
    #left stream     
    x = conv_batchnorm_relu(inputs, filters=filters, kernel_size=1, strides=strides)     
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)     
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)     
    x = BatchNormalization()(x) 
         
    #right stream     
    shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(inputs)     
    shortcut = BatchNormalization()(shortcut)          
    x = Add()([shortcut,x])    #skip connection     
    x = ReLU()(x)          
    return x


def resnet_block(inputs, filters, reps, strides):
    
    x = projection_block(inputs, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
    return x



def Adapted_resnet(i_shape):
    """build the model, but adapting the last layers to upscalling"""
    input_s = Input(shape= i_shape)

    x = conv_batchnorm_relu(input_s, filters=64, kernel_size=3, strides=1)
    x = MaxPool2D(pool_size = 3, strides =2)(x)
    x = resnet_block(x, filters=16, reps =3, strides=1)
    x = resnet_block(x, filters=32, reps =4, strides=2)
    #x = resnet_block(x, filters=256, reps =6, strides=2)
    #x = resnet_block(x, filters=512, reps =3, strides=2)
    #x = GlobalAvgPool2D()(x)
    # Add upscalling part
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    output = Dense(1, activation ='softmax')(x)

    model = Model(inputs=input_s, outputs=output)
    return model









# and https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/


def id_block(x, filters):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filters):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filters, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet34( shape, classes = 1):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = id_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = id_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model