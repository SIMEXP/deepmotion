import tensorflow as tf
import numpy as np
from model_util import *



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


def inference_mocae_mul(input_shape=[None, 22,22,10,1],
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
        
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
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
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
        
        
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
        nfeaturemap = 128
        W = weight_variable([2, 2, 2, 1, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(x, W) + b)
        encoder_i.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
        
    current_input = tf.nn.dropout(current_input, keep_prob, [tf.shape(x)[0],1,1,1,input_nfeaturemap])
    
    with tf.variable_scope('img_conv1_3') as scope:
        shapes_i.append(current_input.get_shape().as_list())
        nfeaturemap = 64
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
        nfeaturemap = 64
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
        nfeaturemap = 128
        W = weight_variable([1, 1, 1, input_nfeaturemap, nfeaturemap])
        b = bias_variable([nfeaturemap])
        output = tf.nn.relu(conv3d(current_input, W) + b)
        encoder_main.append(W)
        input_nfeaturemap = nfeaturemap
        current_input = output
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