import tensorflow as tf
import numpy as np
from model_util import *


def inference_fconv_small12(input_shape=[None, 22,22,10,1],
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
        nfeaturemap = 256
        W = weight_variable([3, 3, 3, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    branch_image = current_input
    '''
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
        nfeaturemap = 128
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
    
    '''
    
    with tf.variable_scope('conv3_1') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 16
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(branch_image, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    # Max pooling
    #current_input = max_pool_2x2(current_input)
    #'''    
    with tf.variable_scope('conv3_2') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 16
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    # store the latent representation
    z = current_input
    z_input_nfeaturemap = input_nfeaturemap
    '''
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    '''
        
    with tf.variable_scope('deconv_i_1') as scope:
        shapes_i.append(z.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('deconv_i_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W) + b
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        y = output
        
        
    with tf.variable_scope('deconv_m_1') as scope:
        shapes_i.append(z.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('deconv_m_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W) + b
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        m_hat = output
    
   
    loss_m = tf.reduce_mean(tf.square(m-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}

def inference_fconv_small(input_shape=[None, 22,22,10,1],
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
        nfeaturemap = 256
        W = weight_variable([2, 2, 2, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
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
        W = weight_variable([2, 2, 2, 3, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(m, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('motion_conv1_3') as scope:
        shapes_m.append(current_input.get_shape().as_list())
        nfeaturemap = 128
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
        nfeaturemap = 16
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
        nfeaturemap = 16
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    # store the latent representation
    z = current_input
    z_input_nfeaturemap = input_nfeaturemap
    '''
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    '''
        
    with tf.variable_scope('deconv_i_1') as scope:
        shapes_i.append(z.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('deconv_i_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W) + b
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        y = output
        
        
    with tf.variable_scope('deconv_m_1') as scope:
        shapes_i.append(z.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('deconv_m_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W) + b
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        m_hat = output
        
    
    
   
    loss_m = tf.reduce_mean(tf.square(m-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}

def inference_fconv(input_shape=[None, 22,22,10,1],
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
        W = weight_variable([3, 3, 3, input_shape[4], nfeaturemap])
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
        W = weight_variable([3, 3, 3, input_shape_m[4], nfeaturemap])
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
    '''
    with tf.variable_scope('img_conv1_1') as scope:
        shapes_i.append(x.get_shape().as_list())
        nfeaturemap = 256
        W = weight_variable([3, 3, 3, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
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
        nfeaturemap = 128
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output

    branch_motion = current_input
    
    #current_input = tf.concat([branch_image, branch_motion], axis=4)
    #input_nfeaturemap = 256
    current_input = tf.multiply(branch_image,branch_motion)
    #print tf.shape(current_input)[-1]
    #tf.shape(current_input)[-1]
    '''
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
    
    with tf.variable_scope('conv3_2') as scope:
        shapes_main.append(current_input.get_shape().as_list())
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
         
        
    # store the latent representation
    z = current_input
    z_input_nfeaturemap = input_nfeaturemap
    '''
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    '''
        
    with tf.variable_scope('deconv_i_1') as scope:
        shapes_i.append(z.get_shape().as_list())
        nfeaturemap = 16
        W = weight_variable([3, 3, 3, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('deconv_i_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W) + b
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        y = output
        
        
    with tf.variable_scope('deconv_m_1') as scope:
        shapes_i.append(z.get_shape().as_list())
        nfeaturemap = 32
        W = weight_variable([3, 3, 3, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(z, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    with tf.variable_scope('deconv_m_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(current_input, W) + b
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        m_hat = output
        
    
    
   
    loss_m = tf.reduce_mean(tf.square(m-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}

def inference_fconv_supercompact(input_shape=[None, 22,22,10,1],
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
        nfeaturemap = 256
        W = weight_variable([3, 3, 3, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 128
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
        nfeaturemap = 128
        W = weight_variable([3, 3, 3, 3, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(m, W) + b)
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])

    branch_motion = current_input
    
    #current_input = tf.concat([branch_image, branch_motion], axis=4)
    #input_nfeaturemap = 256
    current_input = tf.multiply(branch_image,branch_motion)
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
        
    

    # store the latent representation
    z = current_input
    z_input_nfeaturemap = input_nfeaturemap
    '''
    encoder_main.reverse()
    encoder_i.reverse()
    encoder_m.reverse()
    
    shapes_main.reverse()
    shapes_i.reverse()
    shapes_m.reverse()
    '''
        
   
        
    #current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
    with tf.variable_scope('deconv_i_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 1
        W = weight_variable([1, 1, 1, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(z, W) + b
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        y = output
        
        
    with tf.variable_scope('deconv_m_2') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 3
        W = weight_variable([1, 1, 1, z_input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = conv3d(z, W) + b
        encoder_m.append(W)
        input_nfeaturemap = nfeaturemap
        m_hat = output
        
    
    
   
    loss_m = tf.reduce_mean(tf.square(m-m_hat))
    loss_i = tf.reduce_mean(tf.square(t-y))
    cost = loss_i + loss_m

    # %%
    return {'x': x, 't':t, 'm': m, 'm_hat':m_hat, 'y': y, 'cost': cost, 'loss_i':loss_i, 'loss_m':loss_m, 'keep_prob': keep_prob, 'encoder_main':encoder_main, 'encoder_i':encoder_i, 'encoder_m':encoder_m}