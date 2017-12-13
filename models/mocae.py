import tensorflow as tf
import numpy as np
from model_util import *
from model_util import conv3d

def motion_decoder_fc(z_m_hat):
    # TODO
    decode_m=[]
    return output, decode_m
    
def motion_decoder_conv(z_m_hat, channels):
    
    decode_m=[]
    shapes_m=[]
    
    with tf.variable_scope('motion_conv1') as scope:
        shapes_m.append(z_m_hat.get_shape().as_list())
        nfeaturemap = channels[0]
        W = weight_variable([2, 2, 2, 128+64+32, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z_m_hat, W,stride=1) + b)
        decode_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)  
    
    with tf.variable_scope('motion_conv2') as scope:
        shapes_m.append(current_input.get_shape().as_list())
        nfeaturemap = channels[1]
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W,stride=1) + b
        decode_m.append(W)
        #input_nfeaturemap = nfeaturemap
        #m_hat = output
        
    return output, decode_m, shapes_m
    

def conditional_ae(alpha=1.,input_shape=[None, 22,22,10,1],
         input_shape_m=[None, 22,22,10,3],
                n_filters=[1, 32, 32, 32],
                filter_sizes=[3, 2, 3, 2],
                corruption=False):
    """Build the fMRI model.

    Args:
    images: Images.

    
    """
    
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    m = tf.placeholder(
        tf.float32, input_shape_m, name='m')
    t = tf.placeholder(
        tf.float32, input_shape, name='t')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
    
    encoder_i = []
    decoder_i = []
    encoder_m = []
    encoder_main = []
    shapes_main = []
    shapes_i = []
    shapes_m = []
    
    
    # Max pooling
    m_pool1 = max_pool_2x2(m)
    m_pool2 = max_pool_2x2(m_pool1)
    #m_pool3 = max_pool_2x2(m_pool2)
    
    
    current_input = x
    input_nfeaturemap = 1
    #current_input = tf.multiply(current_input, m,)
    
    
    
    #keep_prob=1.
    ### BRANCH 3d images
    with tf.variable_scope('img_conv1_1') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 32
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 64
        W = weight_variable([5, 5, 5, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W, stride=2) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    
        
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)

    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_4') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W, stride=1) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
    
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_5') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W,stride=1) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        

    #m_hat = current_input  
    #tf.nn.conv3d(x, W, strides=[1, 2, 2, 2, 1], padding='SAME')
        
    z_m_hat, z_i = tf.split(current_input, [64,64], axis=4)  
    
    ######################## DECODE MOTION ########################
   
    m_hat, encoder_m, shapes_m = motion_decoder_conv(z_m_hat, channels=[64,3])
        
    ######################## END DECODE MOTION ########################
        
    # store the latent representation
    #z = current_input
    encoder_i.reverse()
    shapes_i.reverse()
    
    encoder_m.reverse()
    shapes_m.reverse()
    
    
    ######################## ENCODE MOTION ########################
    
    with tf.variable_scope('deconv_motion_1') as scope:  
        shapes = shapes_m[0]
        W = encoder_m[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(m_pool1, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        
        
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)

    with tf.variable_scope('deconv_motion_2') as scope:
        shapes = shapes_m[1]
        W = encoder_m[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        z_m = output
    
    ######################## END ENCODE MOTION ########################
    
    current_input = z_m
    #current_input = tf.concat([z_m, z_i], axis=4)
    #current_input = m_pool1
    
    #print shapes_i[0],current_input.get_shape().as_list()
    with tf.variable_scope('deconv1_1') as scope:  
        shapes = shapes_i[0]
        W = encoder_i[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        
        
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,nfeaturemap])
    # resize upsampling
    #current_input = resize_volumes(current_input, 2, 2, 2)
    #current_input = tf.concat([m, current_input], axis=4)
    
        
    with tf.variable_scope('deconv1_2') as scope:
        shapes = shapes_i[1]
        W = encoder_i[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('deconv1_3') as scope:
        shapes = shapes_i[2]
        W = encoder_i[2]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 2, 2, 2, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        

    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,nfeaturemap])
    # resize upsampling
    #current_input = resize_volumes(current_input, 2, 2, 2)
    #current_input = tf.concat([m, current_input], axis=4)
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
    
    
    
    # Fan out branchs   
    with tf.variable_scope('deconv1_4') as scope:  
        shapes = shapes_i[3]
        W = encoder_i[3]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = deconv + b
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    y = current_input
        

    loss_m = tf.reduce_mean(tf.square(m_pool1-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = alpha*loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}

def inference_mocae_mul_ae(alpha=1.,input_shape=[None, 22,22,10,1],
         input_shape_m=[None, 22,22,10,3],
                n_filters=[1, 32, 32, 32],
                filter_sizes=[3, 2, 3, 2],
                corruption=False):
    """Build the fMRI model.

    Args:
    images: Images.

    
    """
    
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    m = tf.placeholder(
        tf.float32, input_shape_m, name='m')
    t = tf.placeholder(
        tf.float32, input_shape, name='t')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
    
    encoder_i = []
    encoder_m = []
    encoder_main = []
    shapes_main = []
    shapes_i = []
    shapes_m = []
    
    
    # Max pooling
    m_pool1 = max_pool_2x2(m)
    m_pool2 = max_pool_2x2(m_pool1)
    #m_pool3 = max_pool_2x2(m_pool2)
    
    
    current_input = x
    input_nfeaturemap = 1
    #current_input = tf.multiply(current_input, m,)
    
    
    
    #keep_prob=1.
    ### BRANCH 3d images
    with tf.variable_scope('img_conv1_1') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 32
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 64
        W = weight_variable([5, 5, 5, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W, stride=2) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    
        
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)

    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_4') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W, stride=1) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
    
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_5') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 256
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W,stride=1) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        

    #m_hat = current_input  
    #tf.nn.conv3d(x, W, strides=[1, 2, 2, 2, 1], padding='SAME')
        
    z_m_hat, z_i = tf.split(current_input, [128+64+32,32], axis=4)  
    
    ######################## DECODE MOTION ########################
    
    with tf.variable_scope('motion_conv1') as scope:
        shapes_m.append(z_m_hat.get_shape().as_list())
        nfeaturemap = 64
        W = weight_variable([2, 2, 2, 128+64+32, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z_m_hat, W,stride=1) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)  
    
    with tf.variable_scope('motion_conv2') as scope:
        shapes_m.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W,stride=1) + b
        encoder_m.append(W)
        #input_nfeaturemap = nfeaturemap
        m_hat = output
        
    ######################## END DECODE MOTION ########################
        
    # store the latent representation
    #z = current_input
    encoder_i.reverse()
    shapes_i.reverse()
    
    encoder_m.reverse()
    shapes_m.reverse()
    
    
    ######################## ENCODE MOTION ########################
    
    with tf.variable_scope('deconv_motion_1') as scope:  
        shapes = shapes_m[0]
        W = encoder_m[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(m_pool1, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        
        
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)

    with tf.variable_scope('deconv_motion_2') as scope:
        shapes = shapes_m[1]
        W = encoder_m[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        z_m = output
    
    ######################## END ENCODE MOTION ########################
    
    current_input = tf.concat([z_m, z_i], axis=4)
    #current_input = m_pool1
    
    #print shapes_i[0],current_input.get_shape().as_list()
    with tf.variable_scope('deconv1_1') as scope:  
        shapes = shapes_i[0]
        W = encoder_i[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        
        
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,nfeaturemap])
    # resize upsampling
    #current_input = resize_volumes(current_input, 2, 2, 2)
    #current_input = tf.concat([m, current_input], axis=4)
    
        
    with tf.variable_scope('deconv1_2') as scope:
        shapes = shapes_i[1]
        W = encoder_i[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('deconv1_3') as scope:
        shapes = shapes_i[2]
        W = encoder_i[2]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 2, 2, 2, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        

    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,nfeaturemap])
    # resize upsampling
    #current_input = resize_volumes(current_input, 2, 2, 2)
    #current_input = tf.concat([m, current_input], axis=4)
    current_input = tf.contrib.layers.batch_norm(current_input,center=True, scale=True)
    
    
    
    # Fan out branchs   
    with tf.variable_scope('deconv1_4') as scope:  
        shapes = shapes_i[3]
        W = encoder_i[3]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = deconv + b
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    y = current_input
        

    loss_m = tf.reduce_mean(tf.square(m_pool1-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = alpha*loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}

def inference_mocae_multibranch(input_shape=[None, 22,22,10,1],
         input_shape_m=[None, 22,22,10,3],
                n_filters=[1, 32, 32, 32],
                filter_sizes=[3, 2, 3, 2],
                corruption=False):
    """Build the fMRI model.

    Args:
    images: Images.

    
    """
    
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    m = tf.placeholder(
        tf.float32, input_shape_m, name='m')
    t = tf.placeholder(
        tf.float32, input_shape, name='t')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
    
    encoder_i = []
    encoder_m = []
    encoder_main = []
    shapes_main = []
    shapes_i = []
    shapes_m = []
    
    #keep_prob=1.
    ### BRANCH 3d images
    with tf.variable_scope('img_conv1_1') as scope:
        shapes_i.append(x.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        img_1 = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    ### BRANCH motion parameters
    with tf.variable_scope('motion_conv1_1') as scope:
        shapes_m.append(m.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, 3, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(m, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        motion_1 = output
        
    
    current_input = tf.multiply(img_1,motion_1)
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 256
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        img_2 = output
      

    
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    # Max pooling
    motion_1 = max_pool_2x2(motion_1)
    input_nfeaturemap = 128
    
    with tf.variable_scope('motion_conv1_3') as scope:
        shapes_m.append(motion_1.get_shape().as_list())
        nfeaturemap = 256
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(motion_1, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        motion_2 = output
        
    
    
    
    # resize upsampling
    motion_2 = resize_volumes(motion_2, 2, 2, 2)
    
    #current_input = tf.concat([branch_image, branch_motion], axis=4)
    #input_nfeaturemap = 512
    current_input = tf.multiply(img_2,motion_2)
    input_nfeaturemap = 256
    #print tf.shape(current_input)[-1]
    #tf.shape(current_input)[-1]
    
    
    
    
    with tf.variable_scope('conv3_1') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    #'''    
    with tf.variable_scope('conv3_2') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    # store the latent representation
    z = current_input
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    
        
    with tf.variable_scope('deconv1_1') as scope:
        shapes = shapes_main[0]
        W = encoder_main[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(z, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    # resize upsampling
    #current_input = resize_volumes(current_input, 2, 2, 2)
        
    with tf.variable_scope('deconv1_2') as scope:
        shapes = shapes_main[1]
        W = encoder_main[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        middle_branch_fmap = nfeaturemap
        middle_branch = output
        
    #middle_branch_i , middle_branch_m = tf.split(middle_branch, num_or_size_splits=2, axis=4)
    
    # Fan out branchs   
    with tf.variable_scope('image_out_deconv1') as scope:  
        shapes = shapes_i[0]
        W = encoder_i[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(middle_branch, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    
    with tf.variable_scope('image_out_deconv') as scope:
        shapes = shapes_i[1]
        W = encoder_i[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        image_output = deconv + b
    
    y=image_output
    
    with tf.variable_scope('motion_out_deconv1') as scope:
        shapes = shapes_m[0]
        W = encoder_m[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(middle_branch, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    with tf.variable_scope('motion_out_deconv') as scope:
        shapes = shapes_m[1]
        W = encoder_m[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        m_hat = deconv + b
        

    loss_m = tf.reduce_mean(tf.square(m-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}


def inference_mocae_mul(alpha=0.01,input_shape=[None, 22,22,10,1],
         input_shape_m=[None, 22,22,10,3],
                n_filters=[1, 32, 32, 32],
                filter_sizes=[3, 2, 3, 2],
                corruption=False):
    """Build the fMRI model.

    Args:
    images: Images.

    
    """
    
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    m = tf.placeholder(
        tf.float32, input_shape_m, name='m')
    t = tf.placeholder(
        tf.float32, input_shape, name='t')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
    
    encoder_i = []
    encoder_m = []
    encoder_main = []
    shapes_main = []
    shapes_i = []
    shapes_m = []
    
    #keep_prob=1.
    ### BRANCH 3d images
    with tf.variable_scope('img_conv1_1') as scope:
        shapes_i.append(x.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([3, 3, 3, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    branch_image = current_input

    ### BRANCH motion parameters
    with tf.variable_scope('motion_conv1_1') as scope:
        shapes_m.append(m.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([3, 3, 3, 3, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(m, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('motion_conv1_3') as scope:
        shapes_m.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output

    branch_motion = current_input
    
    #current_input = tf.concat([branch_image, branch_motion], axis=4)
    #input_nfeaturemap = 128
    current_input = tf.multiply(branch_image,branch_motion)
    #print tf.shape(current_input)[-1]
    #tf.shape(current_input)[-1]
    
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('conv3_1') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    #'''    
    with tf.variable_scope('conv3_2') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    # store the latent representation
    z = current_input
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    
        
    with tf.variable_scope('deconv1_1') as scope:
        shapes = shapes_main[0]
        W = encoder_main[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(z, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    # resize upsampling
    #current_input = resize_volumes(current_input, 2, 2, 2)
        
    with tf.variable_scope('deconv1_2') as scope:
        shapes = shapes_main[1]
        W = encoder_main[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        middle_branch_fmap = nfeaturemap
        middle_branch = output
        
    #middle_branch_i , middle_branch_m = tf.split(middle_branch, num_or_size_splits=2, axis=4)
    
    # Fan out branchs   
    with tf.variable_scope('image_out_deconv1') as scope:  
        shapes = shapes_i[0]
        W = encoder_i[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(middle_branch, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    
    with tf.variable_scope('image_out_deconv') as scope:
        shapes = shapes_i[1]
        W = encoder_i[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        image_output = deconv + b
    
    y=image_output
    
    with tf.variable_scope('motion_out_deconv1') as scope:
        shapes = shapes_m[0]
        W = encoder_m[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(middle_branch, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
        
    with tf.variable_scope('motion_out_deconv') as scope:
        shapes = shapes_m[1]
        W = encoder_m[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        m_hat = deconv + b
        

    loss_m = tf.reduce_mean(tf.square(m-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = alpha*loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}


def inference_mocae_samew_small(input_shape=[None, 22,22,10,1],
         input_shape_m=[None, 22,22,10,3],
                n_filters=[1, 32, 32, 32],
                filter_sizes=[3, 2, 3, 2],
                corruption=False):
    """Build the fMRI model.

    Args:
    images: Images.

    
    """
    
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    m = tf.placeholder(
        tf.float32, input_shape_m, name='m')
    t = tf.placeholder(
        tf.float32, input_shape, name='t')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
    
    encoder_i = []
    encoder_m = []
    encoder_main = []
    shapes_main = []
    shapes_i = []
    shapes_m = []
    
    #keep_prob=1.
    ### BRANCH 3d images
    with tf.variable_scope('img_conv1_1') as scope:
        shapes_i.append(x.get_shape().as_list())
        nfeaturemap = 64
        W = weight_variable([3, 3, 3, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 32
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    branch_image = current_input

    ### BRANCH motion parameters
    with tf.variable_scope('motion_conv1_1') as scope:
        shapes_m.append(m.get_shape().as_list())
        nfeaturemap = 64
        W = weight_variable([3, 3, 3, 3, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(m, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('motion_conv1_3') as scope:
        shapes_m.append(current_input.get_shape().as_list())
        nfeaturemap = 32
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output

    branch_motion = current_input
    
    #current_input = tf.concat([branch_image, branch_motion], axis=4)
    #input_nfeaturemap = 128
    current_input = tf.multiply(branch_image,branch_motion)
    #print tf.shape(current_input)[-1]
    #tf.shape(current_input)[-1]
    
    with tf.variable_scope('conv3_1') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 8
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
    #'''    
    with tf.variable_scope('conv3_2') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 8
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    # store the latent representation
    z = current_input
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    
        
    with tf.variable_scope('deconv1_1') as scope:
        shapes = shapes_main[0]
        W = encoder_main[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(z, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        current_input = output
        
    with tf.variable_scope('deconv1_2') as scope:
        shapes = shapes_main[1]
        W = encoder_main[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        middle_branch_fmap = nfeaturemap
        middle_branch = output
        
    #middle_branch_i , middle_branch_m = tf.split(middle_branch, num_or_size_splits=2, axis=4)

    # Fan out branchs   
    with tf.variable_scope('image_out_deconv1') as scope:  
        shapes = shapes_i[0]
        W = encoder_i[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(middle_branch, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('image_out_deconv') as scope:
        shapes = shapes_i[1]
        W = encoder_i[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        image_output = deconv + b
    
    y=image_output
    
    with tf.variable_scope('motion_out_deconv1') as scope:
        shapes = shapes_m[0]
        W = encoder_m[0]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(middle_branch, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        output = tf.nn.relu(deconv + b)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('motion_out_deconv') as scope:
        shapes = shapes_m[1]
        W = encoder_m[1]
        nfeaturemap = W.get_shape().as_list()[-2]
        b = bias_variable([nfeaturemap])
        deconv = tf.nn.conv3d_transpose(current_input, 
                                        W, 
                                        output_shape=tf.stack([tf.shape(x)[0], shapes[1], shapes[2], shapes[3], shapes[4]]), 
                                        strides=[1, 1, 1, 1, 1], 
                                        padding='SAME')
        m_hat = deconv + b
        

    cost = tf.reduce_mean(tf.square(t-image_output)) + tf.reduce_mean(tf.square(m-m_hat))

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}