from set_mlp import *
import time
import argparse
import numpy as np
import math

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
    xtrain, ytrain, xtest, ytest = load_fashion_mnist_data(10000, 10000)
    xtrain_012, ytrain_012, xtest_012, ytest_012 = load_fashion_mnist_data(nTrainingSamples, nTestingSamples, [3,4,5])
    xtrain_345, ytrain_345, xtest_345, ytest_345 = load_fashion_mnist_data(nTrainingSamples, nTestingSamples, [1,2,3])


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
    getSET = lambda : SET_MLP((xtrain.shape[1], 200, ytrain.shape[1]), (Sigmoid,Sigmoid), epsilon=2)

    set_mlp1 = getSET()
    set_mlp2 = getSET()

    # train SET-MLP
    set_mlp1.fit(xtrain_012, ytrain_012, xtest_012, ytest_012, loss=MSE, epochs=500, batch_size=batchSize, learning_rate=learningRate,
                momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=False,
                save_filename="Results/set_mlp_"+str(nTrainingSamples)+"_training_samples_e"+str(epsilon)+"_rand")

    set_mlp2.fit(xtrain_345, ytrain_345, xtest_345, ytest_345, loss=MSE, epochs=500, batch_size=batchSize,
                 learning_rate=learningRate,
                 momentum=momentum, weight_decay=weightDecay, zeta=zeta, dropoutrate=dropoutRate, testing=False,
                 save_filename="Results/set_mlp_" + str(nTrainingSamples) + "_training_samples_e" + str(epsilon) + "_rand")

    # test SET-MLP
    accuracy, _ = set_mlp1.predict(xtest_012, ytest_012, batch_size=1)
    print("SET1 - data 012: Accuracy of the last epoch on the testing data: ", accuracy)
    accuracy, _ = set_mlp1.predict(xtest_345, ytest_345, batch_size=1)
    print("SET1 - data 345: Accuracy of the last epoch on the testing data: ", accuracy)
    accuracy, _ = set_mlp1.predict(xtest, ytest, batch_size=1)
    print("SET1 - data    : Accuracy of the last epoch on the testing data: ", accuracy)

    accuracy, _ = set_mlp2.predict(xtest_012, ytest_012, batch_size=1)
    print("SET2 - data 012: Accuracy of the last epoch on the testing data: ", accuracy)
    accuracy, _ = set_mlp2.predict(xtest_345, ytest_345, batch_size=1)
    print("SET2 - data 345: Accuracy of the last epoch on the testing data: ", accuracy)
    accuracy, _ = set_mlp2.predict(xtest, ytest, batch_size=1)
    print("SET2 - data    : Accuracy of the last epoch on the testing data: ", accuracy)

    print("\nCreating new SET")
    set_mlp3 = getSET()

    set_mlp3.w[1].data = np.zeros(set_mlp3.w[1].data.shape)
    set_mlp3.w[2].data = np.zeros(set_mlp3.w[2].data.shape)

    for i in range(1, 3):
        print("  Merging layer %d.." % i)

        # First, copy the weights of the layers and convert to COO format
        w1 = set_mlp1.w[i].tocoo()
        w2 = set_mlp2.w[i].tocoo()

        # Extract coordinates, create two lists of [[x, y]]
        c1 = list(zip(w1.row, w1.col))
        c2 = list(zip(w2.row, w2.col))

        # Dictionary with [x, y] -> weight
        stupidDict = {}
        for c in c1:
            stupidDict[c] = set_mlp1.w[i][c]
        for c in c2:
            if(c in stupidDict):
                # Merge two weights at the same coordinate by averaging
                stupidDict[c] = max(stupidDict[c], set_mlp2.w[1][c])
            else:
                stupidDict[c] = set_mlp2.w[1][c]

        # Convert back to list [(weight, [x, y])], containing the weights and coordinates of both layers
        stupidList = [(stupidDict[k], k) for k in stupidDict.keys()]

        # print("Number of weights after merging:", len(stupidList))
        # print(stupidList[:3], stupidList[-3:])

        # Sort list by weight magnitude
        stupidList.sort(key = lambda entry : abs(entry[0]))
        print(stupidList[:3], stupidList[-3:])

        # Keep only the weights with the highest magnitude, thus keeping sparsity the same
        newWeights = stupidList[-w1.nnz:]
        # newWeights = stupidList
        # print("New number of weights:", len(newWeights))
        # print(newWeights[:3], newWeights[-3:])

        # Create new weight matrix in LIL format
        wNew = lil_matrix(w1.shape, dtype=w1.dtype)
        # Fill weight matrix
        for v, c in newWeights:
            wNew[c] = v
        # Put weight matrix into SET, in CSR format
        set_mlp3.w[i] = wNew.tocsr().copy()

        set_mlp3.b[i] = (set_mlp1.b[i] + set_mlp2.b[i]) / 2

    accuracy, _ = set_mlp3.predict(xtest_012, ytest_012, batch_size=1)
    print("SET3 - data 012: Accuracy of the last epoch on the testing data: ", accuracy)
    accuracy, _ = set_mlp3.predict(xtest_345, ytest_345, batch_size=1)
    print("SET3 - data 345: Accuracy of the last epoch on the testing data: ", accuracy)
    accuracy, _ = set_mlp3.predict(xtest, ytest, batch_size=1)
    print("SET3 - data    : Accuracy of the last epoch on the testing data: ", accuracy)

    print("-----------------")