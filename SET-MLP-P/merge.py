import numpy as np
from set_mlp import *

from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

def merge_topk(nn1, nn2, nnTo, logging=False, prune=True):
    log = lambda *args : print(*args) if logging else None

    log("[merge_topk] Merging two networks")

    assert(nn1.w.keys() == nn2.w.keys() == nnTo.w.keys())
    assert(len(nn1.dimensions) == len(nn2.dimensions) == len(nnTo.dimensions))
    for i in range(len(nn1.dimensions)):
        assert(nn1.dimensions[i] == nn2.dimensions[i] == nnTo.dimensions[i])

    nLayers = list(nnTo.w.keys())

    # Get all weights of both layers, sort them
    for iLayer in nLayers:

        mergedWeights = lil_matrix(nn1.w[iLayer].shape, dtype=nn1.w[iLayer].dtype)

        coo1 = nn1.w[iLayer].tocoo()
        coo2 = nn2.w[iLayer].tocoo()

        log("[merge_topk] Layer %d" % iLayer)

        # Networks might not have the same number of weights in the 'same' layer.
        nWeights = (len(coo1.data) + len(coo2.data)) // 2
        log("[merge_topk]   Number of weights : %d" % nWeights)
        log("[merge_topk]   Difference in number of weights : %d" % abs(len(coo2.data) - len(coo2.data)))

        c = np.abs(np.concatenate((coo1.data, coo2.data)))
        c.sort(kind='mergesort')
        threshold = c[-nWeights] if prune else 0.
        log("[merge_topk]   Threshold placed at %0.6f" % threshold)
        log("[merge_topk]   Weights from nnw1 : %d" % np.count_nonzero(threshold < np.abs(coo1.data)))
        log("[merge_topk]   Weights from nnw2 : %d" % np.count_nonzero(threshold < np.abs(coo2.data)))

        for weight, row, col in zip(coo1.data, coo1.row, coo1.col):
            if threshold < abs(weight):
                mergedWeights[row, col] = weight

        for weight, row, col in zip(coo2.data, coo2.row, coo2.col):
            if threshold < abs(weight):
                mergedWeights[row, col] = weight

        nnTo.w[iLayer] = mergedWeights.tocsr()
        nnTo.b[iLayer] = (nn1.b[iLayer] + nn2.b[iLayer]) / 2

def merge_topk_all(*networks, nnTo=None, logging=False, prune=False):
    log = lambda *args, **kwargs : print(*args, **kwargs) if logging else None
    networks = list(networks)
    log("[merge_topk_all] Merging %d networks" % len(networks))

    all_equal = lambda lst : not lst or lst.count(lst[0]) == len(lst)

    # Assert that there is at least one network to merge
    assert(0 < len(networks))
    # Assert that there is a network to merge all other networks in to
    assert(nnTo is not None)
    # Assert that all networks have the same number of layers
    assert(all_equal([len(nn.w.keys()) for nn in networks + [nnTo]]))
    # Assert that at each depth, all layers have the same dimension
    for i in range(len(nnTo.dimensions)):
        assert(all_equal([nn.dimensions[i] for nn in networks + [nnTo]]))

    nNetworks = len(networks)
    nLayers = list(nnTo.w.keys())

    # For each layer, get weights of all networks
    for iLayer in nLayers:
        log("[merge_topk_all] Layer %d" % iLayer)

        mergedWeights = lil_matrix(nnTo.w[iLayer].shape, dtype=nnTo.w[iLayer].dtype)

        # Convert all networks to coo format
        weightsCoo = [nn.w[iLayer].tocoo() for nn in networks]

        # Networks might not have the same number of weights in the 'same' layer.
        nWeights = sum([len(w.data) for w in weightsCoo]) // nNetworks
        log("[merge_topk]   Number of weights : %d" % nWeights)
        log("[merge_topk]   Difference in number of weights : ", end="")
        log(np.array([len(w.data) for w in weightsCoo]) - nWeights)

        c = np.abs(np.concatenate([nn.data for nn in weightsCoo]))
        c.sort(kind='mergesort')
        threshold = c[-nWeights] if prune else 0.
        log("[merge_topk_all]   Total number of weights : %d" % len(c))
        log("[merge_topk_all]   Threshold placed at %0.6f" % threshold)
        for iNetwork in range(nNetworks):
            n = np.count_nonzero(threshold < np.abs(weightsCoo[iNetwork].data))
            log("[merge_topk_all]     Weights from network %d : %d" % (iNetwork, n))

        for iNetwork in range(nNetworks):
            w = weightsCoo[iNetwork]
            for weight, row, col in zip(w.data, w.row, w.col):
                if threshold < abs(weight):
                    mergedWeights[row, col] = weight

        nnTo.w[iLayer] = mergedWeights.tocsr()
        nnTo.b[iLayer] = np.sum([nn.b[iLayer] for nn in networks], axis=0) / nNetworks
