from set_mlp import *
import time
import argparse
import numpy as np
import math
from merge import *

from scipy.sparse import csr_matrix, coo_matrix, dok_matrix, lil_matrix

# Training settings
parser = argparse.ArgumentParser(description='SET Parallel Training ')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=3000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--dropout-rate', type=float, default=0.2, metavar='D',
                    help='Dropout rate')
parser.add_argument('--weight-decay', type=float, default=0.0002, metavar='W',
                    help='Dropout rate')
parser.add_argument('--epsilon', type=int, default=13, metavar='E',
                    help='Sparsity level')
parser.add_argument('--zeta', type=float, default=0.3, metavar='Z',
                    help='It gives the percentage of unimportant connections which are removed and replaced with '
                         'random ones after every epoch(in [0..1])')
parser.add_argument('--no-neurons', type=int, default=1000, metavar='H',
                    help='Number of neurons in the hidden layer')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')

def getCircular(classes):
    return list(zip(classes, classes[1:] + [classes[0]]))

def getDense(classes):
    if len(classes) < 2:
        return []
    a, *b = classes
    combinations = [[a, c] for c in b]
    return combinations + getDense(b)

if __name__ == "__main__":
    args = parser.parse_args()

    #set model parameters
    noHiddenNeuronsLayer = args.no_neurons
    epsilon = args.epsilon
    zeta = args.zeta
    noTrainingEpochs = args.epochs
    batchSize = args.batch_size
    dropoutRate = args.dropout_rate
    learningRate = args.lr
    momentum = args.momentum
    weightDecay = args.weight_decay

    nEpochs = 200

    # Samples per class
    spc = 500

    # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
    getSET = lambda : SET_MLP((784, 200, 10), (Sigmoid, Sigmoid, Sigmoid), epsilon=3)
    trainSet = lambda nn, xtr, ytr, xte, yte : nn.fit(xtr, ytr, xte, yte, loss=MSE, epochs=nEpochs, batch_size=batchSize, learning_rate=learningRate, momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=True)

    classes = [0, 1, 2, 3]
    xtrain, ytrain, xtest, ytest = load_fashion_mnist_data(10000, 10000, classes)

    # Train networks on data subsets
    networks = []
    for c1, c2 in getDense(classes):
        print("[zip] Working on classes %d and %d" % (c1, c2))
        xtr, ytr, xte, yte = load_fashion_mnist_data(spc*2,  spc*2, [c1, c2])
        nn = getSET()
        trainSet(nn, xtr, ytr, xte, yte)
        networks.append([nn, [c1, c2]])

    # Test networks on data subsets and complete dataset
    print("        c1c2   all")
    for nn, [c1, c2] in networks:
        _, _, xte, yte = load_fashion_mnist_data(spc*2, spc*2, [c1, c2])
        acc1, _ = nn.predict(xte, yte, batch_size=1)
        acc2, _ = nn.predict(xtest, ytest, batch_size=1)
        print("%d & %d : %.3f  %.3f" % (c1, c2, acc1, acc2))

    print("\nCreating new SET")
    setMerged = getSET()
    # setMerged.w[1].data.fill(0)
    # setMerged.w[2].data.fill(0)
    # setMerged.b[1].fill(0)
    # setMerged.b[2].fill(0)

    print("\n=== Performance before merging ===")
    acc, _ = setMerged.predict(xtest, ytest, batch_size=1)
    print("setMerged : %.3f" % acc)



    # Train network on all classes
    print("\n=== Performance after training ===")
    sets = load_fashion_mnist_data(spc*len(classes), spc*len(classes), classes)
    trainSet(setMerged, *sets)
    acc, _ = setMerged.predict(xtest, ytest, batch_size=1)
    print("setMerged : %.3f" % acc)

    print("\n=== Performance after training on individual classes ===")
    for i in classes:
        _, _, xte, yte = load_fashion_mnist_data(0, 10000, [i])
        acc, _ = setMerged.predict(xte, yte, batch_size=1)
        print("    class %d : %0.3f" % (i, acc))



    # Merge subnetworks
    print("\n=== Performance after merging ===")
    merge_topk_all(*[n[0] for n in networks], nnTo=setMerged, logging=False, prune=True)
    acc, _ = setMerged.predict(xtest, ytest, batch_size=1)
    print("setMerged : %.3f" % acc)

    print("\n=== Performance after merging on individual classes ===")
    for i in classes:
        _, _, xte, yte = load_fashion_mnist_data(0, 10000, [i])
        acc,  _ = setMerged.predict(xte, yte, batch_size=1)
        print("    class %d : %0.3f" % (i, acc))