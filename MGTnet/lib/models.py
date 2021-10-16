import tensorflow as tf
import scipy.sparse
import numpy as np
import os, time, shutil
from .resnet import img_inference
from .camera_Ti_v1 import Perspective_Camera
from constants import *

class ae(object):
    def __init__(self,
                 adjs,
                 D,
                 U,
                 F,
                 p,
                 nz,
                 nv,
                 Ti_max,
                 Ti_min,
                 Ti_a=0.95,
                 F_0=3,
                 Training=True,
                 num_epochs=20,
                 learning_rate=0.001,
                 batch_size=16,
                 eval_frequency=100,
                 dir_name='',
                 train_data_path='',
                 checkpoint_path=''):

        self.M_0 = adjs[1].shape[0]
        # Store attributes and bind operations.
        self.adjs, self.D, self.U, self.F, self.p, self.nz, self.F_0 = adjs, D, U, F, p, nz, F_0
        self.Ti_max, self.Ti_min, self.Ti_a = Ti_max, Ti_min, Ti_a
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.train_data_path=train_data_path
        self.checkpoint_path=checkpoint_path
        self.Training = Training
        self.encoder_layer = []
        # Build the computational graph.
        self.build_graph()

    def predict(self,
                imgs,
                data,
                means,
                camera_params,
                labels=None,
                sess=None):
        losses = []
        size = data.shape[0]
        predictions = []
        length = imgs.shape[0]
        isVal = False
        if not sess:  # outside call, for test
            sess, _ = self._get_session(sess)  # load checkpoints
        else:  # inside call, for val
            isVal = True
            length = labels.shape[0]
            # length -= length % self.batch_size

        for begin in range(0, length, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_img = np.zeros(
                (self.batch_size, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
            batch_data = np.zeros(
                (self.batch_size, data.shape[1], data.shape[2]))
            batch_mean = np.zeros((self.batch_size, means.shape[1]))
            batch_camera_params = np.zeros(
                (self.batch_size, camera_params.shape[1]))

            batch_img[:end - begin] = imgs[begin:end]
            batch_data[:end - begin] = data[begin:end]
            batch_mean[:end - begin] = means[begin:end]
            batch_camera_params[:end - begin] = camera_params[begin:end]

            feed_dict = {
                self.input_img: batch_img,
                self.input_data: batch_data,
                self.input_means: batch_mean,
                self.id: 0,
                self.camera_params: batch_camera_params
            }

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(
                    (self.batch_size, labels.shape[1], labels.shape[2]))
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.input_labels] = batch_labels
                batch_pred, batch_loss = sess.run(
                    [self.op_prediction, self.op_loss], feed_dict)
                losses.append(batch_loss)
            else:
                batch_pred, batch_xy = sess.run([self.op_prediction, self.xy],
                                                feed_dict)

            if len(predictions) == 0:
                predictions = batch_pred
                project_xy = batch_xy
            else:
                predictions = np.vstack((predictions, batch_pred))
                project_xy = np.vstack((project_xy, batch_xy))

        if isVal:
            return np.mean(losses)  # for val, only return val loss
        if labels is not None:
            return np.array(predictions)[:length], np.mean(losses)
        else:
            return np.array(predictions)[:length], np.array(
                project_xy)[:length]

    def fit(self,
            train_mode='new',
            val_dict=None):  # add camera_params  train_mode

        sess = None
        t_process, t_wall = time.clock(), time.time()
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        path = os.path.join(self._get_path('checkpoints'), 'model')
        lastStep = 0

        if train_mode == 'new':
            print('new  train')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # tf increase GPU memory slowly
            sess = tf.Session(graph=self.graph, config=config)
            shutil.rmtree(self._get_path('summaries'),
                          ignore_errors=True)  # remove dir
            # writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))
            sess.run(self.op_init)  # global initiation
        elif train_mode == 'tune':
            print("tune train")
            sess, lastStep = self._get_session(None)  # load checkpoints

        else:
            print('train_mode input error')
            return

        max_loss = 1000.0
        loss_average = []
        loss1_average = []
        loss2_average = []

        step = 1
        for epoch in range(1, self.num_epochs + 1):
            sess.run(self.iterator.initializer)
            while True:
                try:
                    batch_img, batch_data, batch_camParams, batch_mean, batch_labels = sess.run(
                        self.next_element)
                    batch_img = normilization(batch_img, norm_boundary, imgs_min_max[0],
                                              imgs_min_max[1])
                    batch_data = normilization(batch_data, norm_boundary, smpl_min_max[0],
                                               smpl_min_max[1])
                    batch_labels = normilization(batch_labels, norm_boundary,
                                                 label_min_max[0],
                                                 label_min_max[1])
                    feed_dict = {
                        self.input_img: batch_img,
                        self.input_data: batch_data,
                        self.input_labels: batch_labels,
                        self.input_means: batch_mean,
                        self.id: step,
                        self.camera_params: batch_camParams
                    }

                    tra, loss, loss1, loss2 = sess.run(
                        [self.op_train, self.op_loss, self.loss1, self.loss2],
                        feed_dict)
                    loss_average.append(loss)
                    loss1_average.append(loss1)
                    loss2_average.append(loss2)
                    # Periodical evaluation of the model.
                    if step % self.eval_frequency == 0:
                        if lastStep != 0:
                            print("-----------this train is tune, lastStep is %s-----------" % lastStep)

                        print('step {} epoch {} / {}):'.format(step, epoch, self.num_epochs))
                        loss_mean = np.array(loss_average).mean()
                        print('train loss_average = {:.8f}'.format(loss_mean))
                        print('loss1 = %.6f, loss2=%.6f' %(np.mean(loss1_average), np.mean(loss2_average)))

                        if val_dict:  # start val predict(self, imgs, data, means, camera_params, labels=None, sess=None):
                            val_loss = self.predict(val_dict['imgs'],
                                                    val_dict['smpl'],
                                                    val_dict['means'],
                                                    val_dict['camera_params'],
                                                    val_dict['label'],
                                                    sess=sess)
                            print('val loss_average = {:.8f}'.format(val_loss))

                        print('time: {:.0f}s (wall {:.0f}s)'.format(
                            time.clock() - t_process,
                            time.time() - t_wall))


                        if loss_mean < max_loss:
                            max_loss = loss_mean
                            self.op_saver_ori.save(sess,
                                                   path,
                                                   global_step=step + lastStep)
                        loss_average = []
                        loss1_average = []
                        loss2_average = []

                    step += 1

                except tf.errors.OutOfRangeError:
                    break

        writer.close()
        sess.close()

    # Methods to construct the computational graph.

    def build_graph(self):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            dataset = tf.data.TFRecordDataset(self.train_data_path)
            dataset = dataset.map(_parse_function)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            # dataset = dataset.shuffle(50)
            # iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            self.iterator = iterator
            self.next_element = next_element

            # Inputs.
            with tf.name_scope('inputs'):
                self.input_img = tf.placeholder(tf.float32,
                                                (self.batch_size, 424, 512, 3),
                                                'img')  # [-1,1]
                self.input_data = tf.placeholder(
                    tf.float32, (self.batch_size, self.M_0, self.F_0),
                    'data')  # [-1,1]
                self.input_labels = tf.placeholder(
                    tf.float32, (self.batch_size, self.M_0, self.F_0),
                    'labels')  # [-1,1]
                self.input_means = tf.placeholder(tf.float32,
                                                  (self.batch_size, 3),
                                                  'means')  # [-1,1]
                self.id = tf.placeholder(tf.int32, (), "id")
                # cameraParams as input
                self.camera_params = tf.placeholder(tf.float32,
                                                    (self.batch_size, 16),
                                                    'camera_params')


            op_outputs1, op_outputs2 = self.inference(self.input_img,
                                                      self.input_data,
                                                      reuse=False)

            self.op_loss = self.loss(op_outputs1, op_outputs2,
                                     self.input_labels)
            self.op_train = self.training(self.op_loss, self.learning_rate)
            self.op_prediction = op_outputs2

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
            self.op_saver_ori = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def inference(self, input_img, input_data, reuse=False):

        self.imgfeature = img_inference(input_img,
                                        2,
                                        reuse=reuse,
                                        is_training=self.Training)
        imgfeature = self.imgfeature

        camera = self._get_camera()
        recover_data = self._recover_data(input_data)

        pool_feature = camera.project(
            recover_data,
            self.input_means,
            imgfeature,
            self.id,
            self.camera_params,
            flag=1)  # list object, channels are:16,32,64
        self.xy = camera.xy
        x = tf.reshape(
            tf.concat([
                self.input_data, pool_feature[0], pool_feature[1],
                pool_feature[2]
            ],
                axis=2), [self.batch_size, self.M_0, 18])
        z = self._encode(x, scope="encoder1", reuse=reuse)
        x1 = self._decode(z, scope="decoder1", reuse=reuse)

        self.encoder_layer = []

        x = tf.reshape(
            tf.concat([x1, pool_feature[0], pool_feature[1], pool_feature[2]],
                      axis=2), [self.batch_size, self.M_0, 18])
        z = self._encode(x, scope="encoder2", reuse=reuse)
        x2 = self._decode(z, scope="decoder2", reuse=reuse)

        return x1, x2

    def loss(self, op_outputs1, op_outputs2, labels):
        with tf.name_scope('loss'):
            with tf.name_scope('data_loss'):
                data_loss1 = tf.losses.mean_squared_error(
                    predictions=op_outputs1,
                    labels=labels,
                    reduction=tf.losses.Reduction.MEAN)
                data_loss2 = tf.losses.mean_squared_error(
                    predictions=op_outputs2,
                    labels=labels,
                    reduction=tf.losses.Reduction.MEAN)
            self.loss1 = data_loss1
            self.loss2 = data_loss2
            return data_loss1 + data_loss2

    def training(self, loss, learning_rate):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op_train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            return op_train

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _recover_data(self, data):
        return (data + self.Ti_a) * (self.Ti_max - self.Ti_min) / (
                2 * self.Ti_a) + self.Ti_min


    def _get_camera(self):
        return Perspective_Camera(self.batch_size)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=self.graph, config=config)

            # filename = tf.train.latest_checkpoint(
            #     self._get_path('checkpoints'))
            # print("latest checkpoint is", filename)
            filename=self.checkpoint_path
            self.op_saver.restore(sess, filename)

            lastStep = int(filename[filename.rfind('-') + 1:])
        return sess, lastStep

    def _xavier_variable(self,
                         name,
                         shape,
                         weight_decay=0.0002,
                         initializer=tf.contrib.layers.xavier_initializer(),
                         is_fc_layer=False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        new_variables = tf.get_variable(name,
                                        shape=shape,
                                        initializer=initializer,
                                        regularizer=regularizer)
        return new_variables

    def _weight_variable(self, name, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        return var

    def _bias_variable(self, name, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable(name, shape, tf.float32, initializer=initial)
        return var

    def batch_normalization(self, input):
        return tf.layers.batch_normalization(input, training=self.Training)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin * N])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

        return x

    def fc(self, x, Mout, relu=False):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable("fc_w", [int(Min), Mout],
                                  regularization=True)
        b = self._bias_variable("fc_b", [Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _encode(self, x, scope="encoder", reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            N, Min, Fin = x.get_shape()
            for i in range(len(self.F)):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    with tf.variable_scope('graph_filter'):
                        tmp = []
                        for b in range(self.batch_size):
                            tmp.append(self.adjs[i + 1])
                        x = self.conv2d(
                            x,
                            tf.convert_to_tensor(np.array(tmp),
                                                 dtype=tf.int32), self.F[i])
                        x = tf.nn.relu(self.batch_normalization(x))
                        self.encoder_layer.append(x)
                    if i < 2:
                        with tf.variable_scope('pooling'):
                            x = self.poolwT(x, self.D[i + 1])

            # Fully connected hidden layers.
            # N, M, F = x.get_shape()
            x = tf.reshape(x, [int(N), int(self.p[-1] * self.F[-1])])  # N x MF
            if self.nz:
                with tf.variable_scope('fc'):
                    x = self.fc(x, int(self.nz[0]))  # N x M0
        return x

    def _decode(self, x, scope="decoder", reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            N = x.get_shape()[0]
            # M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-1] * self.F[-1]))  # N x MF
                x = tf.reshape(
                    x, [int(N), int(self.p[-1]),
                        int(self.F[-1])])  # N x M x F

            for i in range(len(self.F) - 1):
                with tf.variable_scope('upconv{}'.format(i + 1)):
                    with tf.variable_scope('unpooling'):
                        x = self.poolwT(x, self.U[-i - 1])
                    with tf.variable_scope('graph_filter'):
                        tmp = []
                        for b in range(self.batch_size):
                            tmp.append(self.adjs[-i - 2])
                        x = tf.concat([x, self.encoder_layer[-i - 2]], axis=2)
                        x = self.conv2d(
                            x,
                            tf.convert_to_tensor(np.array(tmp),
                                                 dtype=tf.int32),
                            self.F[-i - 2])
                        x = tf.nn.relu(self.batch_normalization(x))

            with tf.variable_scope('mesh_all_1'):
                # x = self.poolwT(x, self.U[0])
                tmp = []
                for b in range(self.batch_size):
                    tmp.append(self.adjs[1])
                self.tensor_adj = tf.convert_to_tensor(np.array(tmp),
                                                       dtype=tf.int32)
                x = self.conv2d(x, self.tensor_adj, int(self.F_0))
                # x = tf.tanh(x)
        return x

    def tile_repeat(self, n, repTime):
        '''
        create something like 111..122..2333..33 ..... n..nn
        one particular number appears repTime consecutively.
        This is for flattening the indices.
        '''
        idx = tf.range(n)
        idx = tf.reshape(idx, [-1, 1])  # Convert to a n x 1 matrix.
        idx = tf.tile(
            idx, [1, repTime]
        )  # Create multiple columns, each column has one number repeats repTime
        y = tf.reshape(idx, [-1])
        return y

    def get_slices(self, x, adj):
        batch_size, num_points, in_channels = x.get_shape().as_list()
        batch_size, input_size, K = adj.get_shape().as_list()
        zeros = tf.zeros([batch_size, 1, in_channels], dtype=tf.float32)
        x = tf.concat([zeros, x], 1)
        x = tf.reshape(x, [batch_size * (num_points + 1), in_channels])
        adj = tf.reshape(adj, [batch_size * num_points * K])
        adj_flat = self.tile_repeat(batch_size, num_points * K)
        adj_flat = adj_flat * (num_points + 1)
        adj_flat = adj_flat + adj
        adj_flat = tf.reshape(adj_flat, [batch_size * num_points, K])
        slices = tf.gather(x, adj_flat)
        slices = tf.reshape(slices, [batch_size, num_points, K, in_channels])
        return slices

    def conv2d(self, x, adj, out_channels):
        batch_size, input_size, in_channels = x.get_shape().as_list()
        W = self._weight_variable("conv_wx", [1, in_channels, out_channels])
        b = self._bias_variable("b", [out_channels])
        w_j = self._weight_variable('conv_w_j',
                                    [1, 1, in_channels, out_channels])
        batch_size, input_size, K = adj.get_shape().as_list()
        # Calculate neighbourhood size for each input - [batch_size, input_size, neighbours]
        adj_size = tf.count_nonzero(adj, 2)
        # deal with unconnected points: replace NaN with 0
        non_zeros = tf.not_equal(adj_size, 0)
        adj_size = tf.cast(adj_size, tf.float32)
        adj_size = tf.where(non_zeros, tf.reciprocal(adj_size),
                            tf.zeros_like(adj_size))
        # adj_size = tf.reciprocal(adj_size)
        # [batch_size, input_size, 1]
        adj_size = tf.reshape(adj_size, [batch_size, input_size, 1])

        wx = tf.nn.conv1d(x, W, 1, 'VALID')

        patches = self.get_slices(x, adj)

        patches = tf.nn.conv2d(patches,
                               w_j,
                               strides=[1, 1, 1, 1],
                               padding='VALID')
        patches = tf.reduce_sum(patches, axis=2)
        patches = tf.multiply(adj_size, patches)
        patches = tf.add(patches, wx) + b

        return patches



def _parse_function(example_proto):
    features = {
        "imgs": tf.FixedLenFeature((), tf.string),
        "smpl": tf.FixedLenFeature((), tf.string),
        "camera_params": tf.FixedLenFeature((), tf.string),
        "means": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    imgs = tf.decode_raw(parsed_features['imgs'], tf.float32)
    smpl = tf.decode_raw(parsed_features['smpl'], tf.float32)
    camera_params = tf.decode_raw(parsed_features['camera_params'], tf.float32)
    means = tf.decode_raw(parsed_features['means'], tf.float32)
    label = tf.decode_raw(parsed_features['label'], tf.float32)
    return tf.reshape(imgs, [-1, 512, 3]), tf.reshape(smpl, [19019, 9]), camera_params, means, \
           tf.reshape(label, [19019, 9])


def normilization(data, a, min_, max_):
    return 2 * a * (data - min_) / (max_ - min_) - a


def recover_normilization(data, a, min_, max_):
    return (data + a) * (max_ - min_) / (2 * a) + min_
