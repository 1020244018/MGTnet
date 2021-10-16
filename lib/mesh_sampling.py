import math
import heapq
import numpy as np
import os
import scipy.sparse as sp
from psbody.mesh import Mesh
from opendr.topology import get_vert_connectivity, get_vertices_per_edge


def generate_transform_matrices(mesh_path, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.
    
    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """

    assert len(factors) == 3
    M,A,D,U = [], [], [], []

    # for mesh up up 
    mesh = Mesh(filename=os.path.join(mesh_path,"smpl_mesh_up_up.obj"))
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)

    # for mesh up  
    mesh = Mesh(filename=os.path.join(mesh_path,"smpl_mesh_up.obj"))
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)
    D.append(sp.eye(19019,61718))
    U.append(sp.load_npz(os.path.join(mesh_path,"up_sampling_stage_two.npz")))#saprse shape=(61718, 19019)

    # for smpl  
    mesh = Mesh(filename=os.path.join(mesh_path,"smpl_mesh.obj"))
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)
    D.append(sp.eye(6890,19019))
    U.append(sp.load_npz(os.path.join(mesh_path,"up_sampling_stage_one.npz")))#sparse shape=(19019, 6890)

    # for smpl down
    mesh = Mesh(filename=os.path.join(mesh_path,"smpl_mesh_down.obj"))
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)
    D.append(sp.load_npz(os.path.join(mesh_path, "down_sampling_1723_6890.npz")))#sparse shape=1723x6890
    U.append(sp.load_npz(os.path.join(mesh_path, "up_sampling_6890_1723.npz")))#sparse shape=6890x1723

    return M,A,D,U

