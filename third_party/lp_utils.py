import os
import sys
import io
import time
import numpy as np 
import scipy

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, minres, qmr





import platform
if platform.system() == "Windows":
    sys.path.append(os.path.join(os.path.dirname(__file__), './kgraph-win/build/Release/'))
import pykgraph


import contextlib

@contextlib.contextmanager
def time_debug(name, suppress_output=False):
    if suppress_output:
        sys.stdout = io.StringIO()
    start = time.time()
    yield
    end = time.time()
    if suppress_output:
        sys.stdout = sys.__stdout__

    print("%s : cost %0.4fs"%(name, end-start))


def sm_affi(features, k, method='kgraph', metric='euclidean', kernel=lambda x:np.exp(-x)):
    """
        data的相似度稀疏矩阵（n×n）
    """
    # row = []
    # col = []
    # data = []

    assert method in ['scipy', 'kgraph']
    assert metric in ['euclidean', 'angular']

    total_size = features.shape[0]

    if method == 'scipy':
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1).fit(features)
        A = kneighbors_graph(nbrs, k, mode='distance', include_self=False, n_jobs=-1)
        distance_var = np.sqrt(np.mean(np.square(A.data)))
        A.data = kernel(A.data / distance_var)
        
    elif method == 'kgraph':
        # build kgraph
        index = pykgraph.KGraph(features.astype(np.float32), metric)  # another option is 'angular'
        index.build()

        # search 
        ind, dists = index.search(features, K=k+1, withDistance=True)

        # exclude self
        ind = ind[:, 1:].reshape([-1])
        dists = dists[:, 1:].reshape([-1]) * 2.0

        distances = np.sqrt(dists)
        distance_var = np.sqrt(np.mean(dists))

        # 
        data = kernel(distances / distance_var)
        row, col = np.repeat(np.arange(total_size), k, axis=0), ind
        A = csr_matrix((data, (row, col)), 
                shape=(total_size, total_size))


    W = A + A.transpose()
    D = W.sum(axis=1).reshape(-1)
    D = scipy.sparse.spdiags(D, 0, total_size, total_size).power(-0.5)

    # 归一化权重矩阵
    W = D * W * D

    return W



def sm_label(label, label_indices, total_size, nb_classes):
    row = label_indices
    col = label
    data = [1 for _ in label_indices]
    return csr_matrix((data, (row, col)), shape=(total_size, nb_classes))



def conjugate_gradient_solver(A, b, solver='cg'):
    res = []

    solver_func = {
        'bicg' : bicg, 'bicgstab' : bicgstab, 'cg' : cg, 'cgs' : cgs, 'gmres' : gmres, 'minres' : minres, 'qmr' : qmr,
    }.get(solver, None)
    assert solver_func != None

    for i in range(b.shape[1]):
        if isinstance(b, np.ndarray):
            X, info = solver_func(A=A, b=b[:,i])
        else:
            X, info = solver_func(A=A, b=b[:,i].toarray())
        if int(info) != 0:
            print("Warning : info is not zero %f"%info)
        res.append(X)
    return np.array(res).transpose([1, 0])




def jacobi_iteration_solver(A, b):

    def jacobi(A,b,N=25,x=None):
        """Solves the equation Ax=b via the Jacobi iterative method."""
        # Create an initial guess if needed                                                                                                                                                            
        if x is None:
            x = np.zeros(A.shape[0])

        # Create a vector of the diagonal elements of A                                                                                                                                                
        # and subtract them from A                                                                                                                                                                     
        D = np.diag(A)
        R = A - np.diagflat(D)

        # Iterate for N times                                                                                                                                                                          
        for i in range(N):
            x = (b - np.dot(R,x)) / D
        return x

    res = []
    for i in range(b.shape[1]):
        if isinstance(b, np.ndarray):
            X = jacobi(A=A, b=b[:,i])
        else:
            X = jacobi(A=A, b=b[:,i].toarray())
        res.append(X)
    return np.array(res).transpose([1, 0])



def entropy_weight(Z):
    Z = np.maximum(Z, 5e-10)
    nb_classes = Z.shape[1]
    Z = Z / Z.sum(axis=1, keepdims=True)
    Z = - 1.0 *  (Z * np.log(Z)).sum(axis=1)
    W = 1.0 - Z / np.log(float(nb_classes))
    return W


def graph_laplace(feature_list, label, label_indices, nb_classes, n=10, alpha=0.99, **kwargs):

    with time_debug("graph_laplace"):
        with time_debug("\tsm_affi,sm_label"):
            A = sm_affi(feature_list, n, **kwargs)
            b = sm_label(label, label_indices, feature_list.shape[0], nb_classes)
            I = scipy.sparse.eye(feature_list.shape[0])

        with time_debug('\tcg_solver'):
            Y0 = conjugate_gradient_solver(I - A * alpha, b)
            W0 = entropy_weight(Y0)
            Y0 = np.argmax(Y0, axis=1)

        # for s in ['bicg', 'bicgstab', 'cgs', 'gmres', 'minres', 'qmr']:
        #     with time_debug('\t%s_solver'%s):
        #         Y1 = conjugate_gradient_solver(I - A * alpha, b, solver=s)
        #         W1 = entropy_weight(Y1)
        #         Y1 = np.argmax(Y1, axis=1)
        #     print(float((Y0 == Y1).astype(np.int32).sum()) / float(len(feature_list)))

        # with time_debug('\tji_solver'):
        #     Y1 = jacobi_iteration_solver(I - A * alpha, b)
        #     W1 = entropy_weight(Y1)
        #     Y1 = np.argmax(Y1, axis=1)

    return Y0, W0



if __name__ == "__main__":

    import pickle as pkl
    import time

    # test_pkl_filepath = "./feature_list_250.pkl"

    # feature, emb_feature = pkl.load(open(test_pkl_filepath, 'rb'))

    # start = time.time()
    # W = sm_affi(feature, 10)
    # end = time.time()
    # print(start - end)

    # start = time.time()
    # W = sm_affi2(feature, 10)
    # end = time.time()

    # print(start - end)

