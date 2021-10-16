# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
from psbody.mesh import Mesh
from scipy.sparse import *
from scipy.sparse import linalg as sla


def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


class Perspective_Camera():
    # Input is in numpy array format
    def __init__(self, batch_size=1):
        self.generate_L(batch_size)
        self.rec = Mesh(filename='./lib/camera_T/template.obj')

    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.cast(tf.SparseTensor(indices, coo.data, coo.shape), dtype=tf.float32)

    # points: Nx3
    def transform(self, points, trans, rotm):
        points = tf.transpose(points, [0, 2, 1])
        points = tf.matmul(rotm, points)
        points = tf.transpose(points, [0, 2, 1])
        res = points + trans

        return res

    # Solve the sparse linear system ||LX-B||^2 question, get X
    # L.T*LX=L.T*B
    def generate_L(self, batch_size):
        self.regressor = np.load('./lib/camera_T/J_regressor.npy')
        self.wij = np.sqrt(np.load("./lib/camera_T/Wij.npy"))
        self.edge_pair = np.load("./lib/camera_T/Edgeij.npy")
        self.eij = np.load("./lib/camera_T/eij.npy")
        stack_eij = np.multiply(np.stack((self.eij, self.eij, self.eij), axis=1),
                                np.repeat(self.wij, 9).reshape([-1, 3, 3]))
        self.wij_eij = tf.convert_to_tensor(np.tile(stack_eij, [batch_size, 1, 1]).reshape([-1, 3, 3]),
                                            dtype=tf.float32)
        wij_L = np.stack((self.wij, -self.wij), axis=-1).flatten()
        row = np.stack((list(range(self.edge_pair.shape[0])), list(range(self.edge_pair.shape[0]))), axis=-1).flatten()
        self.L = coo_matrix((wij_L, (row, self.edge_pair.flatten()))).tocsr()
        A = self.L.transpose().dot(self.L)
        # self.factor = cholesky(A)
        self.factor = sla.splu(A)
        self.L = self.convert_sparse_matrix_to_sparse_tensor(self.L.transpose())

    def generate_B(self, Tis, mean, id, flag):
        Tis = tf.gather(Tis, self.edge_pair[:, 0], axis=1)
        batch, edge, _ = Tis.shape
        Tis = tf.reshape(Tis, [batch * edge, 3, 3])

        B = tf.reduce_sum(tf.multiply(Tis, self.wij_eij), axis=-1)
        B = tf.reshape(B, [batch, edge, 3])

        b_x = tf.transpose(tf.squeeze(B[:, :, 0]))
        b_y = tf.transpose(tf.squeeze(B[:, :, 1]))
        b_z = tf.transpose(tf.squeeze(B[:, :, 2]))

        b_x = tf.sparse_tensor_dense_matmul(self.L, b_x)
        b_y = tf.sparse_tensor_dense_matmul(self.L, b_y)
        b_z = tf.sparse_tensor_dense_matmul(self.L, b_z)

        vertices = tf.py_func(self.solve_B, [b_x, b_y, b_z, mean, id, flag, batch], tf.float32)

        return vertices

    def solve_B(self, b_x, b_y, b_z, mean, id, flag, batch):

        x = self.factor.solve(b_x).transpose()
        y = self.factor.solve(b_y).transpose()
        z = self.factor.solve(b_z).transpose()
        meshes = np.stack((x, y, z), axis=2)

        centroid_A = np.mean(meshes, axis=1)
        t = mean - centroid_A

        vertice_number = meshes.shape[1]
        t = np.tile(np.reshape(t, [batch, -1, 3]), (1, vertice_number, 1))
        meshes = meshes + t
        return np.array(meshes).astype(np.float32)


    def meshloss(self, vertices_T, mean, id, flag, gt_mesh):
        vertices = self.generate_B(vertices_T, mean, id, flag)
        mesh_loss = tf.losses.mean_squared_error(predictions=vertices, labels=gt_mesh,
                                                          reduction=tf.losses.Reduction.MEAN)
        return mesh_loss

    # Point is a Tensor
    def project(self, vertices_T, mean, img_feat, id, camera_params, flag):
        batch_size, num_points, in_channels = vertices_T.get_shape().as_list()
        vertices = self.generate_B(vertices_T, mean, id, flag)

        points = vertices + 1e-8

        # load camera_params
        fl_x, fl_y, = tf.expand_dims(camera_params[:, 0], -1), tf.expand_dims(camera_params[:, 1], -1)
        cx, cy = tf.expand_dims(camera_params[:, 2], -1), tf.expand_dims(camera_params[:, 3], -1)
        trans, rotm = tf.reshape(camera_params[:, 4:7], (-1, 1, 3)), tf.reshape(camera_params[:, 7:], (
        -1, 3, 3))

        points = self.transform(points, trans, rotm)
        xs = tf.divide(points[:, :, 0], points[:, :, 2])
        ys = tf.divide(points[:, :, 1], points[:, :, 2])
        us = fl_x * xs + cx  # w
        vs = fl_y * ys + cy  # h
        us = tf.minimum(tf.maximum(us, 0), 2 * cx - 1)
        vs = tf.minimum(tf.maximum(vs, 0), 2 * cy - 1)

        x = vs
        y = us
        self.xy = tf.concat([tf.expand_dims(x, -1), tf.expand_dims(y, -1)], axis=2)


        out1 = self.mesh_img(img_feat[0], x, y, img_feat[0].shape[-1], num_points)

        x = vs
        y = us
        out2 = self.mesh_img(img_feat[1], x, y, img_feat[1].shape[-1], num_points)
        x = vs
        y = us
        out3 = self.mesh_img(img_feat[2], x, y, img_feat[2].shape[-1], num_points)
        return [out1, out2, out3]

    def mesh_img(self, img_feat, x, y, dim, num_points):
        x1 = tf.floor(x)
        x2 = tf.ceil(x)
        y1 = tf.floor(y)
        y2 = tf.ceil(y)
        a = tf.stack([tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 2)
        idx = tf.range(img_feat.shape[0])
        idx = tf.reshape(idx, [-1, 1])
        idx = tf.tile(idx, [1, num_points])
        Q11 = tf.gather_nd(img_feat,
                           tf.stack([tf.cast(idx, tf.int32), tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 2))

        Q12 = tf.gather_nd(img_feat,
                           tf.stack([tf.cast(idx, tf.int32), tf.cast(x1, tf.int32), tf.cast(y2, tf.int32)], 2))

        Q21 = tf.gather_nd(img_feat,
                           tf.stack([tf.cast(idx, tf.int32), tf.cast(x2, tf.int32), tf.cast(y1, tf.int32)], 2))

        Q22 = tf.gather_nd(img_feat,
                           tf.stack([tf.cast(idx, tf.int32), tf.cast(x2, tf.int32), tf.cast(y2, tf.int32)], 2))

        weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y2, y))
        weights = tf.expand_dims(weights, 2)
        Q11 = tf.multiply(weights, Q11)


        weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y2, y))
        weights = tf.expand_dims(weights, 2)
        Q21 = tf.multiply(weights, Q21)


        weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y, y1))
        weights = tf.expand_dims(weights, 2)
        Q12 = tf.multiply(weights, Q12)

        weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y, y1))
        weights = tf.expand_dims(weights, 2)
        Q22 = tf.multiply(weights, Q22)

        outputs = tf.add_n([Q11, Q21, Q12, Q22])
        return outputs


def compute_mean(mesh):
    res = []
    for m in mesh:
        centroid_A = np.mean(m, axis=0)
        res.append(centroid_A)
    return np.array(res)
