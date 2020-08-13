###############################################################################
#
# File: evaluate_unflow.py
#
# Primitive hacking of things from eval_gui.py in the UnFlow package to
# allow running evaluations
#
# History:
# 07-30-20 - Levi Burner - Created file
#
###############################################################################

import os
import sys

import numpy as np
import tensorflow as tf

from motion_illusions.UnFlow.src.e2eflow.core.flow_util import flow_to_color, flow_error_avg, outlier_pct
from motion_illusions.UnFlow.src.e2eflow.core.flow_util import flow_error_image
from motion_illusions.UnFlow.src.e2eflow.util import config_dict
from motion_illusions.UnFlow.src.e2eflow.core.image_warp import image_warp
from motion_illusions.UnFlow.src.e2eflow.core.unsupervised import unsupervised_loss
from motion_illusions.UnFlow.src.e2eflow.core.input import resize_input, resize_output_crop, resize_output, resize_output_flow
from motion_illusions.UnFlow.src.e2eflow.core.train import restore_networks
from motion_illusions.UnFlow.src.e2eflow.ops import forward_warp
from motion_illusions.UnFlow.src.e2eflow.core.losses import DISOCC_THRESH, occlusion
from motion_illusions.UnFlow.src.e2eflow.util import convert_input_strings

def load_experiment_config_override_directories(override_dirs, override_res, exp_name):
    # Construct the directory the experiment resides in
    exp_dir = os.path.join(override_dirs['log'], 'ex', exp_name)

    # Load the experiments configuration to retrieve the training parameters
    exp_config_path = os.path.join(exp_dir, 'config.ini')
    if not os.path.isfile(exp_config_path):
        raise RuntimeError('Experiment directory must contain config file')

    exp_config = config_dict(exp_config_path)
    exp_train_params = exp_config['train']

    # Interpret the training parameters while overriding the directories used
    # when loading fine-tuned checkpoints
    convert_input_strings(exp_train_params, override_dirs)

    # Hack in the resolution we are evaluating
    exp_train_params.update({'height': override_res[0], 'width': override_res[1]})

    ckpt_state = tf.train.get_checkpoint_state(exp_dir)
    if not ckpt_state:
        raise RuntimeError("Error: experiment must contain a checkpoint")
    ckpt_path = os.path.join(exp_dir, os.path.basename(ckpt_state.model_checkpoint_path))

    return (exp_train_params, ckpt_state, ckpt_path)

def construct_graph(data_input, exp_train_params):
    inputs = data_input.input()

    im1, im2, input_shape = inputs[:3]
    truth = inputs[3:]

    # Evaluating the loss function actually computes the flow as well
    _, flow, flow_bw = unsupervised_loss(
        (im1, im2),
        normalization=data_input.get_normalization(),
        params=exp_train_params, augment=False, return_flow=True)

    height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)

    im1_pred = image_warp(im2, flow)
    im1_diff = tf.abs(im1 - im1_pred)

    if len(truth) == 4:
        flow_occ, mask_occ, flow_noc, mask_noc = truth
        flow_occ = resize_output_crop(flow_occ, height, width, 2)
        flow_noc = resize_output_crop(flow_noc, height, width, 2)
        mask_occ = resize_output_crop(mask_occ, height, width, 1)
        mask_noc = resize_output_crop(mask_noc, height, width, 1)

        #div = divergence(flow_occ)
        #div_bw = divergence(flow_bw)
        occ_pred = 1 - (1 - occlusion(flow, flow_bw)[0])
        def_pred = 1 - (1 - occlusion(flow, flow_bw)[1])
        disocc_pred = forward_warp(flow_bw) < DISOCC_THRESH
        disocc_fw_pred = forward_warp(flow) < DISOCC_THRESH
        image_slots = [((im1 * 0.5 + im2 * 0.5) / 255, 'overlay'),
                       (im1_diff / 255, 'brightness error'),
                       #(im1 / 255, 'first image', 1, 0),
                       #(im2 / 255, 'second image', 1, 0),
                       #(im2_diff / 255, '|first - second|', 1, 2),
                       (flow_to_color(flow), 'flow'),
                       #(flow_to_color(flow_bw), 'flow bw prediction'),
                       #(tf.image.rgb_to_grayscale(im1_diff) > 20, 'diff'),
                       #(occ_pred, 'occ'),
                       #(def_pred, 'disocc'),
                       #(disocc_pred, 'reverse disocc'),
                       #(disocc_fw_pred, 'forward disocc prediction'),
                       #(div, 'div'),
                       #(div < -2, 'neg div'),
                       #(div > 5, 'pos div'),
                       #(flow_to_color(flow_occ, mask_occ), 'flow truth'),
                       (flow_error_image(flow, flow_occ, mask_occ, mask_noc),
                        'flow error') #  (blue: correct, red: wrong, dark: occluded)
        ]

        # list of (scalar_op, title)
        scalar_slots = [(flow_error_avg(flow_noc, flow, mask_noc), 'EPE_noc'),
                        (flow_error_avg(flow_occ, flow, mask_occ), 'EPE_all'),
                        (outlier_pct(flow_noc, flow, mask_noc), 'outliers_noc'),
                        (outlier_pct(flow_occ, flow, mask_occ), 'outliers_all')]
    elif len(truth) == 2:
        flow_gt, mask = truth
        flow_gt = resize_output_crop(flow_gt, height, width, 2)
        mask = resize_output_crop(mask, height, width, 1)

        image_slots = [((im1 * 0.5 + im2 * 0.5) / 255, 'overlay'),
                       (im1_diff / 255, 'brightness error'),
                       (flow_to_color(flow), 'flow'),
                       (flow_to_color(flow_gt, mask), 'gt'),
        ]

        # list of (scalar_op, title)
        scalar_slots = [(flow_error_avg(flow_gt, flow, mask), 'EPE_all')]
    else:
        image_slots = [(im1 / 255, 'first image'),
                       #(im1_pred / 255, 'warped second image', 0, 1),
                       (im1_diff / 255, 'warp error'),
                       #(im2 / 255, 'second image', 1, 0),
                       #(im2_diff / 255, '|first - second|', 1, 2),
                       (flow_to_color(flow), 'flow prediction')]
        scalar_slots = []
        print('no truth')

    num_ims = len(image_slots)
    image_ops = [t[0] for t in image_slots]
    scalar_ops = [t[0] for t in scalar_slots]
    image_names = [t[1] for t in image_slots]
    scalar_names = [t[1] for t in scalar_slots]
    all_ops = image_ops + scalar_ops

    return num_ims, flow, flow_bw, all_ops, image_ops, image_names, scalar_ops, scalar_names

def evaluate_graph(sess,
                   exp_name, exp_train_params, ckpt_state, ckpt_path,
                   num_ims, flow, flow_bw, all_ops,
                   scalar_names, num_frames_to_evaluate):
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    restore_networks(sess, exp_train_params, ckpt_state, ckpt_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,
                                           coord=coord)

    # TODO adjust for batch_size > 1 (also need to change image_lists appending)
    image_lists = []
    averages = np.zeros(len(scalar_names))
    max_iter = num_frames_to_evaluate #FLAGS.num if FLAGS.num > 0 else None

    try:
        num_iters = 0
        while not coord.should_stop() and (max_iter is None or num_iters != max_iter):
            all_results = sess.run([flow, flow_bw] + all_ops)
            flow_fw_res, flow_bw_res = all_results[:2]
            all_results = all_results[2:]
            image_results = all_results[:num_ims]
            scalar_results = all_results[num_ims:]
            iterstr = str(num_iters).zfill(6)

            image_lists.append(image_results)
            averages += scalar_results
            if num_iters > 0:
                sys.stdout.write('\r')
            num_iters += 1
            sys.stdout.write("-- evaluating '{}': {}/{}"
                             .format(exp_name, num_iters, max_iter))
            sys.stdout.flush()
            print()
    except tf.errors.OutOfRangeError:
        pass

    averages /= num_iters

    coord.request_stop()
    coord.join(threads)

    return image_lists, averages 

def evaluate_experiment(override_dirs, exp_name, data_input, num_frames_to_evaluate):
    (exp_train_params,
     ckpt_state,
     ckpt_path) = load_experiment_config_override_directories(override_dirs, data_input.dims, exp_name)

    with tf.Graph().as_default():
        (num_ims, flow, flow_bw, all_ops, image_ops, image_names, scalar_ops, scalar_names) = construct_graph(data_input, exp_train_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            (image_lists, averages) = evaluate_graph(sess,
                                                     exp_name, exp_train_params, ckpt_state, ckpt_path,
                                                     num_ims, flow, flow_bw, all_ops, scalar_names, num_frames_to_evaluate)

    for scalar_name, avg in zip(scalar_names, averages):
        print("({}) {} = {}".format(name, scalar_name, avg))

    return image_lists, image_names
