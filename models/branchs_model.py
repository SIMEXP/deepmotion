import tensorflow as tf
import numpy as np
from model_util import *


def inference_2obj(input_shape=[None, 22,22,10,1],
         input_shape_m=[None, 22,22,10,3],
                n_filters=[1, 32, 32, 32],
                filter_sizes=[3, 2, 3, 2],
                corruption=False):
    """Build the fMRI model.

    Args:
    images: Images returned from distorted_inputs() or inputs().

    Returns:
    Logits.
    """
    
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    m = tf.placeholder(
        tf.float32, input_shape_m, name='m')
    t = tf.placeholder(
        tf.float32, input_shape, name='t')
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
    #keep_prob=1.
    ### BRANCH 3d images
    with tf.variable_scope('img_conv1_1') as scope:
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
       
    
    '''
    with tf.variable_scope('img_conv1_2') as scope:
        nfeaturemap = 32
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
    '''
    # Max pooling
    current_input = max_pool_2x2(current_input)


    with tf.variable_scope('img_conv2_1') as scope:
        nfeaturemap = 256
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    #'''    
    with tf.variable_scope('img_conv2_2') as scope:
        nfeaturemap = 512
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
    #'''    
    
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv2_3') as scope:
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = resize_volumes(current_input, 2, 2, 2)

    '''
    with tf.variable_scope('img_deconv1') as scope:
        nfeaturemap = 512
        W = weight_variable([3, 3, 3, nfeaturemap, input_nfeaturemap])
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], nfeaturemap]), 
                                        strides=[1, 2, 2, 2, 1], 
                                        padding='SAME',
                                       name='Deconvolution1')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
    '''
    #current_input = tf.contrib.keras.layers.UpSampling3D(current_input, size=2)
    
    #'''
    branch_image = current_input

    ### BRANCH motion parameters
    with tf.variable_scope('motion_conv1_1') as scope:
        nfeaturemap = 64
        W = weight_variable([2, 2, 2, 3, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(m, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    
    # Max pooling
    #current_input = max_pool_2x2(current_input)

    with tf.variable_scope('motion_conv1_2') as scope:
        nfeaturemap = 64
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('motion_conv1_3') as scope:
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output

    branch_motion = current_input
    
    current_input = tf.multiply(branch_image, branch_motion)
    

    with tf.variable_scope('conv3_1') as scope:
        nfeaturemap = 512
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #'''
    with tf.variable_scope('conv3_2') as scope:
        nfeaturemap = 256
        W = weight_variable([3, 3, 3, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
    #'''    
    with tf.variable_scope('conv3_3') as scope:
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('conv3_4') as scope:
        nfeaturemap = 64
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('conv3_5') as scope:
        nfeaturemap = 32
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        input_nfeaturemap = nfeaturemap
        current_input_bis = output

    with tf.variable_scope('image_out_conv3') as scope:
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input_bis, W) + b
        
    y=output
    
    with tf.variable_scope('motion_out_conv3') as scope:
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        m_hat = conv3d(current_input_bis, W) + b

    

    # cost function
    cost = tf.reduce_mean(tf.square(y - t)) + tf.reduce_mean(tf.square(m - m_hat))

    # %%
    return {'x': x, 't': t, 'm': m, 'y': y, 'm_hat':m_hat, 'cost': cost, 'keep_prob': keep_prob}

