import numpy as np
import tensorflow as tf
from .. import registration


def avgpool3d(hr_vols, k=3):
    init = tf.global_variables_initializer()
    x = tf.placeholder(tf.float32, shape=(hr_vols.shape[0], hr_vols.shape[1], hr_vols.shape[2], hr_vols.shape[3], 1))
    y = tf.nn.avg_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1], padding='SAME')

    with tf.Session() as sess:
        results = sess.run(y, feed_dict={x: hr_vols[:, ...][..., np.newaxis]})

    lr_vols = results[..., 0]
    return lr_vols


def apply_signal(template, N=1000, mu_gm=80., sigma_gm=0.05, binary_threshold=0.2):
    ref_vol = template.copy()
    # sigma is a % of the basline mu
    sigma = sigma_gm * mu_gm
    wm = np.ones((N,)) * 70.
    csf = np.ones((N,)) * 100.
    ext = np.ones((N,)) * 0.

    # init vols
    vols = np.zeros((template.shape[0], template.shape[1], template.shape[2], N))

    # put signal in the vols for each class
    vols[ref_vol == 0] = ext
    vols[ref_vol == 1] = wm
    vols[ref_vol == 3] = csf
    print vols[ref_vol == 2].shape
    # vols[ref_vol==2] = 2.*sigma*(np.random.randn(*(vols[ref_vol==2].shape))>.2)+mu_gm
    vols[ref_vol == 2] = 2. * sigma * (np.random.randn((N)) > binary_threshold) + mu_gm
    # vols[ref_vol==2] = gm

    # re-organize the dims
    vols = np.swapaxes(vols[np.newaxis, ...], 0, -1)[..., 0]

    return vols

def apply_motion(vols, v2w, motion_params):
    # generate a simulation of N time points
    vols_motion = []
    for i in range(len(motion_params)):
        coreg_vol, transf = registration.transform(vols[i, ...], motion_params[i], v2w)
        vols_motion.append(coreg_vol)
    vols_motion = np.stack(vols_motion)
    return vols_motion

def object_generator(shape, w_size=[], stride=[6, 6, 6], center=[], obj_type='2square'):
    x, y, z = w_size
    if center == []:
        center = (np.array(shape) / 2.).astype(int) - 1
    vol = np.zeros(shape)
    print shape, center

    v2w = np.eye(4)
    v2w[:3, 3] = -center
    # print w_size,h_size

    if obj_type == '2square':
        print center - x / 2
        vol[center[0] - x / 2:center[0] + x / 2, \
        center[1] - y / 2:center[1] + y / 2, \
        center[2] - z / 2:center[2] + z / 2] = 1

        vol[stride[0] + center[0] - x / 2:-stride[0] + center[0] + x / 2, \
        stride[1] + center[1] - y / 2:-stride[1] + center[1] + y / 2, \
        stride[2] + center[2] - z / 2:-stride[2] + center[2] + z / 2] = 3

        vol[stride[0] + center[0] - x / 3:-stride[0] + center[0] + x / 3, \
        center[1] - y / 2:-stride[1] * 3 + center[1] + y / 2, \
        stride[2] + center[2] - z / 3:-stride[2] + center[2] + z / 3] = 2

    return vol, v2w

