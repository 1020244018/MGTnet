import os
import numpy as np
from psbody.mesh import Mesh
from scipy.sparse import coo_matrix

from scipy.sparse import linalg as sla


# Solve the sparse linear system ||LX-B||^2 question, get X
# L.T*LX=L.T*B

def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def generateL():
    wij = np.sqrt(np.load("./lib/camera_T/Wij.npy"))  # shape = (12954,)
    edge_pair = np.load("./lib/camera_T/Edgeij.npy")  # shape = (12954, 2)
    wij = np.stack((wij, -wij), axis=-1).flatten()
    row = np.stack((list(range(edge_pair.shape[0])), list(range(edge_pair.shape[0]))), axis=-1).flatten()
    L = coo_matrix((wij, (row, edge_pair.flatten()))).tocsr()
    return L

def generateB(Ti, wij, eid, edge):
    Ti = np.take(Ti, edge_pair[:, 0], axis=0)
    Ti = Ti.reshape([-1, 3, 3])
    eij_ = np.stack((eij, eij, eij), axis=1)
    B = np.sum((Ti * eij_), axis=-1)
    b_x = wij * B[:, 0]
    b_y = wij * B[:, 1]
    b_z = wij * B[:, 2]
    return b_x, b_y, b_z

L = generateL()
A = L.transpose().dot(L)
factor = sla.splu(A)

wij = np.sqrt(np.load("./lib/camera_T/Wij.npy"))
eij = np.load("./lib/camera_T/eij.npy")
edge_pair = np.load("./lib/camera_T/Edgeij.npy")


def convert2obj(pred, means, savefolder):
    os.makedirs(savefolder, exist_ok=True)
    for i in range(pred.shape[0]):
        b_x, b_y, b_z = generateB(pred[i], wij, eij, edge_pair)
        b_x = L.transpose().dot(np.array(b_x)).reshape((-1, 1))
        b_y = L.transpose().dot(np.array(b_y)).reshape((-1, 1))
        b_z = L.transpose().dot(np.array(b_z)).reshape((-1, 1))

        x = factor.solve(b_x)
        y = factor.solve(b_y)
        z = factor.solve(b_z)

        p = np.stack((x, y, z), axis=1).squeeze()
        reference = Mesh(filename='./lib/camera_T/template.obj')

        centroid = np.mean(p, axis=0)
        t = centroid - means[i]
        p -= t
        write_mesh_as_obj(os.path.join(savefolder, str(i + 1).zfill(4) + '.obj'), p, reference.f)
