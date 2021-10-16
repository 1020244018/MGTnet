import numpy as np
import json
import os
import copy
import argparse
import cv2
from constants import *
from lib import models, utils, mesh_sampling
from test_reconstruct import convert2obj

parser = argparse.ArgumentParser(
    description='Tensorflow Trainer for Convolutional Mesh Autoencoders')
parser.add_argument('--name',
                    default='doublefusion',
                    help='facial_motion| lfw ')
parser.add_argument('--data',
                    default='data/train/',
                    help='facial_motion| lfw ')
parser.add_argument('--batch_size',
                    type=int,
                    default=10,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs',
                    type=int,
                    default=200,
                    help='number of epochs to train (default: 2)')
parser.add_argument('--eval_frequency',
                    type=int,
                    default=200,
                    help='eval frequency')
parser.add_argument('--nz',
                    type=int,
                    default=128,
                    help='Size of latent variable')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--seed',
                    type=int,
                    default=2,
                    help='random seed (default: 1)')
parser.add_argument('--mode', default='test', type=str, help='train or test')
parser.add_argument('--viz', type=int, default=0, help='visualize while test')
parser.add_argument('--mesh_path',
                    default='lib/sampling_mesh',
                    help='visualize while test')
parser.add_argument('--train_mode', default='new', help='new or tune')
parser.add_argument('--val_mode', default=True, help=True or False)
parser.add_argument('--train_data_path', default='data/train/df4305outlier.tfrecords', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints/doublefusion/model-53700', type=str)

args = parser.parse_args()
np.random.seed(args.seed)

ds_factors = [4, 4, 4]  # Sampling factor of the mesh at each stage of sampling

# Generates adjecency matrices A, downsampling matrices D, and upsamling matrices U by sampling
# the mesh 4 times. Each time the mesh is sampled by a factor of 4
M, A, D, U = mesh_sampling.generate_transform_matrices(
    args.mesh_path, ds_factors)

A = [x.astype('float32') for x in A]
adjacency = utils.get_adjs(A)

D = [x.astype('float32') for x in D]
U = [x.astype('float32') for x in U]
p = [x.shape[0] for x in A]

params = dict()
params['dir_name'] = args.name
params['num_epochs'] = args.num_epochs
params['batch_size'] = args.batch_size
params['eval_frequency'] = args.eval_frequency
params['Ti_min'] = smpl_min_max[0]
params['Ti_max'] = smpl_min_max[1]
params['Ti_a'] = norm_boundary

# Architecture.
params['F_0'] = 9  # Number of graph input features.
params['F'] = [16, 16, 16]  # Number of graph convolutional filters.
params['p'] = p  # Pooling sizes.
params['nz'] = [args.nz]  # Output dimensionality of fully connected layers.

# Optimization.
params['nv'] = 19019  # n_vertex
params['learning_rate'] = args.lr
params['Training'] = True if args.mode == "train" else False

# Path
params['train_data_path']=args.train_data_path
params['checkpoint_path']=args.checkpoint_path

model = models.ae(adjs=adjacency, D=D, U=U, **params)


def normilization(data, a, min_, max_):
    return 2 * a * (data - min_) / (max_ - min_) - a

def recover_normilization(data, a, min_, max_):
    return (data + a) * (max_ - min_) / (2 * a) + min_


if args.mode in ['test']:
    # for testdata
    test_path = './data/test/'
    img = np.load(test_path + "imgs.npy")
    smpl_up = np.load(test_path + 'smpl.npy')
    means = np.load(test_path + 'means.npy')
    camera_params = np.load(test_path + 'cams.npy')

    imgs = normilization(img, norm_boundary, imgs_min_max[0], imgs_min_max[1])
    smpl_up_up_Ti = normilization(smpl_up, norm_boundary, smpl_min_max[0], smpl_min_max[1])
    if not os.path.exists('predict'):
        os.makedirs('predict')

    pred, project_xy = model.predict(imgs, smpl_up_up_Ti, means, camera_params)

    print("sample nums:", imgs.shape[0])
    print("pred.shape=", pred.shape)
    pred = np.array(pred)
    pred = recover_normilization(pred, norm_boundary, label_min_max[0], label_min_max[1])

    # save result
    save_root='predict'
    np.save(os.path.join(save_root, "predict_result_test.npy"), pred)
    np.save(os.path.join(save_root, "project_xy_test.npy"), project_xy)


    # mesh result
    savefolder=os.path.join(save_root, 'mesh')
    convert2obj(pred, means, savefolder)

    # project result
    save_image_path = os.path.join(save_root,'project')
    os.makedirs(save_image_path, exist_ok=True)
    img = np.load(test_path + "imgs.npy")
    for i in range(img.shape[0]):
        cur_img = img[i]
        cur_project_mask = np.zeros((cur_img.shape[0], cur_img.shape[1]))
        cur_project_point = project_xy[i]
        for img_x, img_y in cur_project_point:
            cur_project_mask[int(img_x), int(img_y)] = 1
        cur_project_mask = np.expand_dims(cur_project_mask, -1)
        fuse = cur_img * (1 - cur_project_mask) + (
                cur_img * cur_project_mask * 0.5 +
                np.array([120, 0, 0]).reshape([1, 1, 3]))
        cv2.imwrite(save_image_path + '/' + str(i) + '.png', fuse)

else:
    print("--------------------- training preparing ---------------------")
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name))
        print("create dir checkpoints/" + args.name)
    with open(os.path.join('checkpoints', args.name + 'params.json'),
              'w') as fp:
        saveparams = copy.deepcopy(params)
        saveparams['seed'] = args.seed
        json.dump(saveparams, fp)
        print("save params")

    val_dict=None
    train_mode = args.train_mode
    if train_mode == 'new':
        print("--------------------- start new training ---------------------")
        t_step = model.fit(val_dict=val_dict)
        train_mode = 'tune'
    elif train_mode == 'tune':
        print("--------------------- start tune training ---------------------")
        t_step = model.fit(train_mode='tune', val_dict=val_dict)
