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

if __name__ == "__main__":
    args = parser.parse_args()


    ### Load datasets
    nTrainingSamples = 1000
    nTestingSamples = 1000
    xtrain, ytrain, xtest, ytest = load_fashion_mnist_data(10000, 10000, [1, 2, 3])
    xtrain_12, ytrain_12, xtest_12, ytest_12 = load_fashion_mnist_data(nTrainingSamples, nTestingSamples, [1, 2])
    xtrain_23, ytrain_23, xtest_23, ytest_23 = load_fashion_mnist_data(nTrainingSamples, nTestingSamples, [2, 3])
    xtrain_13, ytrain_13, xtest_13, ytest_13 = load_fashion_mnist_data(nTrainingSamples, nTestingSamples, [1, 3])

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


    # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
    getSET = lambda : SET_MLP((784, 200, 10), (Sigmoid, Sigmoid, Sigmoid), epsilon=3)

    set12 = getSET()
    set23 = getSET()
    set13 = getSET()

    nEpochs = 200

    # train SET-MLP
    set12.fit(xtrain_12, ytrain_12, xtest_12, ytest_12, loss=MSE, epochs=nEpochs, batch_size=batchSize, learning_rate=learningRate,
                momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=True)

    set23.fit(xtrain_23, ytrain_23, xtest_23, ytest_23, loss=MSE, epochs=nEpochs, batch_size=batchSize, learning_rate=learningRate,
                 momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=True)

    set13.fit(xtrain_13, ytrain_13, xtest_13, ytest_13, loss=MSE, epochs=nEpochs, batch_size=batchSize, learning_rate=learningRate,
              momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=True)

    # test SET-MLP
    print("\n\n=============== Testing networks ===============")
    print("        Train Test")
    accuracy1, _ = set12.predict(xtest_12, ytest_12, batch_size=1)
    accuracy2, _ = set12.predict(xtest, ytest, batch_size=1)
    print("SET12 - %0.2f  %0.2f " % (accuracy1, accuracy2))

    accuracy1, _ = set23.predict(xtest_23, ytest_23, batch_size=1)
    accuracy2, _ = set23.predict(xtest, ytest, batch_size=1)
    print("SET23 - %0.2f  %0.2f " % (accuracy1, accuracy2))

    accuracy1, _ = set13.predict(xtest_13, ytest_13, batch_size=1)
    accuracy2, _ = set13.predict(xtest, ytest, batch_size=1)
    print("SET13 - %0.2f  %0.2f " % (accuracy1, accuracy2))

    print("\nCreating new SET")
    set123 = getSET()
    set123.w[1].data.fill(0)
    set123.w[2].data.fill(0)
    set123.b[1].fill(0)
    set123.b[2].fill(0)

    print("\n=== Performance before merging ===")
    accuracy12, _ = set123.predict(xtest_12, ytest_12, batch_size=1)
    accuracy23, _ = set123.predict(xtest_23, ytest_23, batch_size=1)
    accuracy13, _ = set123.predict(xtest_13, ytest_13, batch_size=1)
    accuracy123,_ = set123.predict(xtest, ytest, batch_size=1)
    print("SET123 -  12 %0.2f" % accuracy12)
    print("SET123 -  23 %0.2f" % accuracy23)
    print("SET123 -  13 %0.2f" % accuracy13)
    print("SET123 - 123 %0.2f" % accuracy123)

    merge_topk_all(set12, set23, set13, nnTo=set123, logging=False, prune=True)

    print("\n=== Performance after merging ===")
    accuracy12, _ = set123.predict(xtest_12, ytest_12, batch_size=1)
    accuracy23, _ = set123.predict(xtest_23, ytest_23, batch_size=1)
    accuracy13, _ = set123.predict(xtest_13, ytest_13, batch_size=1)
    accuracy123,_ = set123.predict(xtest, ytest, batch_size=1)
    print("SET123 -  12 %0.2f" % accuracy12)
    print("SET123 -  23 %0.2f" % accuracy23)
    print("SET123 -  13 %0.2f" % accuracy13)
    print("SET123 - 123 %0.2f" % accuracy123)

    for i in [1, 2, 3]:
        trainX, trainY, testX, testY = load_fashion_mnist_data(10000, 10000, [i])
        accuracyTrain, _ = set123.predict(trainX, trainY, batch_size=1)
        accuracyTest,  _ = set123.predict(testX, testY, batch_size=1)
        print("    class %d : train %0.2f %0.2f test" % (i, accuracyTrain, accuracyTest))

    exit()