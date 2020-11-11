# Authors: Decebal Constantin Mocanu et al.;
# Code associated with ECMLPKDD 2019 tutorial "Scalable Deep Learning: from theory to practice"; https://sites.google.com/view/sdl-ecmlpkdd-2019-tutorial
# This is a pre-alpha free software and was tested in Ubuntu 16.04 with Python 3.5.2, Numpy 1.14, SciPy 0.19.1, and (optionally) Cython 0.27.3;

# If you use parts of this code please cite the following article:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#If you have space please consider citing also these articles

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

#@article{Liu2019onemillion,
#  author =        {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#  journal =       {arXiv:1901.09181},
#  title =         {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#  year =          {2019},
#}

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
#the "sparseoperations" Cython library was tested in Ubuntu 16.04. Please note that you may encounter some "solvable" issues if you compile it in Windows.
import sparseoperations
import datetime
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# from keras.datasets import cifar10
# from keras.utils import np_utils
# from keras.preprocessing.image import ImageDataGenerator


def backpropagation_updates_Numpy(a, delta, rows, cols, out):
    for i in range(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createSparseWeights(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float64(np.random.randn() / 10)
    # print("Create sparse matrix with ", weights.getnnz(), " connections and ",
    #       (weights.getnnz() / (noRows * noCols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def array_intersect(A, B):
    # this are for array intersection
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype))  # boolean return


class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))



class MSE:
    def __init__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class NoActivation:
    """
    This is a plugin function for no activation.

    f(x) = x * 1
    """

    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        return z

    @staticmethod
    def prime(z):
        """
        The prime of z * 1 = 1
        :param z: (array)
        :return: z': (array)
        """
        return np.ones_like(z)


class SET_MLP:
    def __init__(self, dimensions, activations, epsilon=20):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes


        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------

        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.droprate = 0  # dropout rate
        self.dimensions = dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}
        self.activations = {}

        t1 = datetime.datetime.now()

        for i in range(len(dimensions) - 1):
            self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i], dimensions[i + 1]) #create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

        # with  ProcessPoolExecutor() as executor:
        #     # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        #     self.activations = {}
        #     noRows = dimensions[:-1]
        #     noCols = dimensions[1:]
        #
        #     # create sparse weight matrices
        #
        #     results = executor.map(createSparseWeights, [epsilon] * len(noCols), noRows, noCols)
        #     for i, res in enumerate(results):
        #         self.w[i + 1] = res
        #         self.b[i + 1] = np.zeros(dimensions[i + 1])
        #         self.activations[i + 2] = activations[i]
        t2 = datetime.datetime.now()

        # print("Creation sparse weights time: ", t2 - t1)

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            if not drop:
                if (i > 1):
                    z[i + 1] = z[i + 1] * (1 - self.droprate)
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if (drop):
                if (i < self.n_layers - 1):
                    dropMask = np.random.rand(a[i + 1].shape[0], a[i + 1].shape[1])
                    dropMask[dropMask >= self.droprate] = 1
                    dropMask[dropMask < self.droprate] = 0
                    a[i + 1] = dropMask * a[i + 1]

        return z, a

    def _back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = coo_matrix(self.w[self.n_layers - 1])

        # compute backpropagation updates
        sparseoperations.backpropagation_updates_Cython(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)
        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        # backpropagation_updates_Numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(), delta)
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
            dw = coo_matrix(self.w[i - 1])

            # compute backpropagation updates
            sparseoperations.backpropagation_updates_Cython(a[i - 1], delta, dw.row, dw.col, dw.data)
            # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            # backpropagation_updates_Numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(), delta)
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if (index not in self.pdw):
            self.pdw[index] = -self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * np.mean(delta, 0)
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * np.mean(delta, 0)

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropoutrate=0, testing=True, save_filename=""):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        :param momentum: (flt)
        :param weight_decay: (flt)
        :param zeta: (flt) #control the fraction of weights removed
        :param droprate: (flt)
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.droprate = dropoutrate
        self.save_filename = save_filename
        self.batchSize = batch_size
        self.inputLayerConnections = []
        self.inputLayerConnections.append(self.getCoreInputConnections())
        np.savez_compressed(self.save_filename + "_input_connections.npz",
                            inputLayerConnections=self.inputLayerConnections)

        maximum_accuracy = 0

        metrics = np.zeros((epochs, 4))

        for i in range(epochs):

            log = "[fit] Epoch %d" % i

            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            # training
            t1 = datetime.datetime.now()

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l], True)
                self._back_prop(z, a, y_[k:l])

            t2 = datetime.datetime.now()

            # log += "; training: " + str(t2 - t1)

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test, self.batchSize)
                accuracy_train, activations_train = self.predict(x, y_true, self.batchSize)
                t4 = datetime.datetime.now()

                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test

                log += "; accuracy train: %0.4f" % accuracy_train
                log += "; accuracy test: %0.4f" % accuracy_test

                # print("Testing time: ", t4 - t3,"; Loss train: ", loss_train, "; Loss test: ", loss_test, "; Accuracy train: ", accuracy_train,"; Accuracy test: ", accuracy_test,
                #       "; Maximum accuracy test: ", maximum_accuracy)

            t5 = datetime.datetime.now()
            if (i < epochs - 1):  # do not change connectivity pattern after the last epoch
                # self.weightsEvolution_I() #this implementation is more didactic, but slow.
                self.weightsEvolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()

            # log += "; evolution: " + str(t6 - t5)

            # save performance metrics values in a file
            if (self.save_filename != ""):
                np.savetxt(self.save_filename+".txt", metrics)

            print("\r%s" % log, end="")
        print(" - Complete")
        return metrics

    def getCoreInputConnections(self):
        values = np.sort(self.w[1].data)
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)

        largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
        smallestPositive = values[
            int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        wlil = self.w[1].tolil()
        wdok = dok_matrix((self.dimensions[0], self.dimensions[1]), dtype="float64")

        # remove the weights closest to zero
        keepConnections = 0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if ((val < largestNegative) or (val > smallestPositive)):
                    wdok[ik, jk] = val
                    keepConnections += 1
        return wdok.tocsr().getnnz(axis=1)

    def weightsEvolution_I(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers):

            values = np.sort(self.w[i].data)
            firstZeroPos = find_first_pos(values, 0)
            lastZeroPos = find_last_pos(values, 0)

            largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
            smallestPositive = values[
                int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float64")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float64")

            # remove the weights closest to zero
            keepConnections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if ((val < largestNegative) or (val > smallestPositive)):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keepConnections += 1

            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keepConnections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.randn() / 10
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weightsEvolution_II(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        #evolve all layers, except the one from the last hidden layer to the output layer
        for i in range(1, self.n_layers):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            #if(self.w[i].count_nonzero()/(self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8):
                t_ev_1 = datetime.datetime.now()
                # converting to COO form
                wcoo = self.w[i].tocoo()
                valsW = wcoo.data
                rowsW = wcoo.row
                colsW = wcoo.col

                pdcoo = self.pdw[i].tocoo()
                valsPD = pdcoo.data
                rowsPD = pdcoo.row
                colsPD = pdcoo.col
                # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
                values = np.sort(self.w[i].data)
                firstZeroPos = find_first_pos(values, 0)
                lastZeroPos = find_last_pos(values, 0)

                largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
                smallestPositive = values[
                    int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

                # remove the weights (W) closest to zero and modify PD as well
                valsWNew = valsW[(valsW > smallestPositive) | (valsW < largestNegative)]
                rowsWNew = rowsW[(valsW > smallestPositive) | (valsW < largestNegative)]
                colsWNew = colsW[(valsW > smallestPositive) | (valsW < largestNegative)]

                newWRowColIndex = np.stack((rowsWNew, colsWNew), axis=-1)
                oldPDRowColIndex = np.stack((rowsPD, colsPD), axis=-1)

                newPDRowColIndexFlag = array_intersect(oldPDRowColIndex, newWRowColIndex)  # careful about order

                valsPDNew = valsPD[newPDRowColIndexFlag]
                rowsPDNew = rowsPD[newPDRowColIndexFlag]
                colsPDNew = colsPD[newPDRowColIndexFlag]

                self.pdw[i] = coo_matrix((valsPDNew, (rowsPDNew, colsPDNew)),
                                         (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                if(i==1):
                    self.inputLayerConnections.append(coo_matrix((valsWNew, (rowsWNew, colsWNew)),
                                       (self.dimensions[i - 1], self.dimensions[i])).getnnz(axis=1))
                    np.savez_compressed(self.save_filename + "_input_connections.npz",
                                        inputLayerConnections=self.inputLayerConnections)

                # add new random connections
                keepConnections = np.size(rowsWNew)
                lengthRandom = valsW.shape[0] - keepConnections
                randomVals = np.random.randn(lengthRandom) / 10
                zeroVals = 0 * randomVals  # explicit zeros

                # adding  (wdok[ik,jk]!=0): condition
                while (lengthRandom > 0):
                    ik = np.random.randint(0, self.dimensions[i - 1], size=lengthRandom, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=lengthRandom, dtype='int32')

                    randomWRowColIndex = np.stack((ik, jk), axis=-1)
                    randomWRowColIndex = np.unique(randomWRowColIndex, axis=0)  # removing duplicates in new rows&cols
                    oldWRowColIndex = np.stack((rowsWNew, colsWNew), axis=-1)

                    uniqueFlag = ~array_intersect(randomWRowColIndex, oldWRowColIndex)  # careful about order & tilda

                    ikNew = randomWRowColIndex[uniqueFlag][:, 0]
                    jkNew = randomWRowColIndex[uniqueFlag][:, 1]
                    # be careful - row size and col size needs to be verified
                    rowsWNew = np.append(rowsWNew, ikNew)
                    colsWNew = np.append(colsWNew, jkNew)

                    lengthRandom = valsW.shape[0] - np.size(rowsWNew)  # this will constantly reduce lengthRandom

                # adding all the values along with corresponding row and column indices
                valsWNew = np.append(valsWNew, randomVals)
                # valsPDNew=np.append(valsPDNew, zeroVals)
                if (valsWNew.shape[0] != rowsWNew.shape[0]):
                    print("not good")
                self.w[i] = coo_matrix((valsWNew, (rowsWNew, colsWNew)),
                                       (self.dimensions[i - 1], self.dimensions[i])).tocsr()

                # print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

                t_ev_2 = datetime.datetime.now()
                # print("Weights evolution time for layer",i,"is", t_ev_2 - t_ev_1)

    def predict(self, x_test, y_test, batch_size=1):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test = self._feed_forward(x_test[k:l])
            activations[k:l] = a_test[self.n_layers]
        correctClassification = 0
        for j in range(y_test.shape[0]):
            if (np.argmax(activations[j]) == np.argmax(y_test[j])):
                correctClassification += 1
        accuracy = correctClassification / y_test.shape[0]
        return accuracy, activations

def load_fashion_mnist_data(noTrainingSamples, noTestingSamples, classes=None, seed=None):
    # If seed is None, then RandomState will try to read data from /dev/urandom
    # (or the Windows analogue) if available or seed from the clock otherwise.
    np.random.seed(seed)

    if classes is None:
        classes = list(range(10)) # Activate all classes

    # Load data
    data = np.load("../Tutorial-IJCAI-2019-Scalable-Deep-Learning/data/fashion_mnist.npz")

    # Get active classes, one-hot encoded
    activeClasses = np.eye(10)[:, ::-1][classes].tolist()
    # Function to check if class is active
    isActive = lambda oneHotRow : oneHotRow.tolist() in activeClasses

    activeIndicesTrain = np.where([isActive(row) for row in data["Y_train"]])[0]
    activeIndicesTest  = np.where([isActive(row) for row in data["Y_test"]])[0]

    np.random.shuffle(activeIndicesTrain)
    np.random.shuffle(activeIndicesTest)

    X_train = data["X_train"][activeIndicesTrain[0:noTrainingSamples], :]
    Y_train = data["Y_train"][activeIndicesTrain[0:noTrainingSamples], :]
    X_test = data["X_test"][activeIndicesTest[0:noTestingSamples], :]
    Y_test = data["Y_test"][activeIndicesTest[0:noTestingSamples], :]

    # normalize in 0..1
    X_train = X_train.astype('float64') / 255.
    X_test = X_test.astype('float64') / 255.

    return X_train, Y_train, X_test, Y_test

# def load_cifar10_data(noTrainingSamples,noTestingSamples):
#     # np.random.seed(0)
#
#     # read CIFAR10 data
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
#     y_train = np_utils.to_categorical(y_train, 10)
#     y_test = np_utils.to_categorical(y_test, 10)
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#
#     indexTrain = np.arange(x_train.shape[0])
#     np.random.shuffle(indexTrain)
#
#     indexTest = np.arange(x_test.shape[0])
#     np.random.shuffle(indexTest)
#
#     x_train = x_train[indexTrain[0:noTrainingSamples], :]
#     y_train = y_train[indexTrain[0:noTrainingSamples], :]
#     x_test = x_test[indexTest[0:noTestingSamples], :]
#     y_test = y_test[indexTest[0:noTestingSamples], :]
#
#     # normalize data
#     xTrainMean = np.mean(x_train, axis=0)
#     xTtrainStd = np.std(x_train, axis=0)
#     x_train = (x_train - xTrainMean) / xTtrainStd
#     x_test = (x_test - xTrainMean) / xTtrainStd
#
#     x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float64')
#     x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float64')
#     return x_train, y_train, x_test, y_test


def image_data_augmentation(x_train):
    print('Using real-time data augmentation.')

    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Return data generator
    # Usage: datagen.flow(x_train, y_train, batch_size=batch_size)
    return datagen
