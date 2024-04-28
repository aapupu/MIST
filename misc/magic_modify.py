"""
# MAGIC was modified based on
# https://github.com/dpeerlab/magic/blob/published/src/magic/MAGIC_core.py
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix, find
from sklearn.neighbors import NearestNeighbors

def magic_modify(adata, latent, t=3, k=30, ka=10, epsilon=1, rescale=99, noise=1):

    L = compute_markov(data=adata.obsm[latent], k=k, epsilon=epsilon, 
                       distance_metric='euclidean', ka=ka)
    new_data, L_t = impute_fast(adata.raw.X.toarray(), L, t, rescale_percent=rescale)
    new_data[new_data <= noise] = 0
    
    return new_data

def impute_fast(data, L, t, rescale_percent=0, L_t=None, tprev=None):

    #convert L to full matrix
    if issparse(L):
        L = L.todense()

    #L^t
    print('MAGIC: L_t = L^t')
    if L_t == None:
        L_t = np.linalg.matrix_power(L, t)
    else:
        L_t = np.dot(L_t, np.linalg.matrix_power(L, t-tprev))

    print('MAGIC: data_new = L_t * data')
    data_new = np.array(np.dot(L_t, data))

    #rescale data
    if rescale_percent != 0:
        if len(np.where(data_new < 0)[0]) > 0:
            print('Rescaling should not be performed on log-transformed '
                  '(or other negative) values. Imputed data returned unscaled.')
            return data_new, L_t
            
        M99 = np.percentile(data, rescale_percent, axis=0)
        M100 = data.max(axis=0)
        indices = np.where(M99 == 0)[0]
        M99[indices] = M100[indices]
        M99_new = np.percentile(data_new, rescale_percent, axis=0)
        M100_new = data_new.max(axis=0)
        indices = np.where(M99_new == 0)[0]
        M99_new[indices] = M100_new[indices]
        max_ratio = np.divide(M99, M99_new)
        data_new = np.multiply(data_new, np.tile(max_ratio, (len(data), 1)))
    
    return data_new, L_t

def compute_markov(data, k=10, epsilon=1, distance_metric='euclidean', ka=0):

    N = data.shape[0]

    # Nearest neighbors
    print('Computing distances')
    nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(data)
    distances, indices = nbrs.kneighbors(data)

    if ka > 0:
        print('Autotuning distances')
        for j in reversed(range(N)):
            temp = sorted(distances[j])
            lMaxTempIdxs = min(ka, len(temp))
            if lMaxTempIdxs == 0 or temp[lMaxTempIdxs] == 0:
                distances[j] = 0
            else:
                distances[j] = np.divide(distances[j], temp[lMaxTempIdxs])

    # Adjacency matrix
    print('Computing kernel')
    rows = np.zeros(N * k, dtype=np.int32)
    cols = np.zeros(N * k, dtype=np.int32)
    dists = np.zeros(N * k)
    location = 0
    for i in range(N):
        inds = range(location, location + k)
        rows[inds] = indices[i, :]
        cols[inds] = i
        dists[inds] = distances[i, :]
        location += k
    if epsilon > 0:
        W = csr_matrix( (dists, (rows, cols)), shape=[N, N] )
    else:
        W = csr_matrix( (np.ones(dists.shape), (rows, cols)), shape=[N, N] )

    # Symmetrize W
    W = W + W.T

    if epsilon > 0:
        # Convert to affinity (with selfloops)
        rows, cols, dists = find(W)
        rows = np.append(rows, range(N))
        cols = np.append(cols, range(N))
        dists = np.append(dists/(epsilon ** 2), np.zeros(N))
        W = csr_matrix( (np.exp(-dists), (rows, cols)), shape=[N, N] )

    # Create D
    D = np.ravel(W.sum(axis = 1))
    D[D!=0] = 1/D[D!=0]

    #markov normalization
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W)

    return T
