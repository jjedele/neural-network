import numpy as np

# activation functions
def sigmoid(x):
    """Sigmoid/logistic activation function."""
    value = 1.0 / (1 + np.exp(-x))
    gradient = value * (1 - value)
    return value, gradient

def tanh(x):
    """Hyperbolic tangent activation function."""
    value = np.tanh(x)
    gradient = 1.0 - value**2
    return value, gradient

def relu(x):
    """Restricted linear unit activation function."""
    value = np.maximum(0.0, x)
    gradient = np.zeros_like(x)
    gradient[x > 0] = 1.0
    return value, gradient

def softmax(x):
    """Softmax activation function.
    
    Softmax is applied per row."""
    row_max = np.max(x, axis=1, keepdims=True)
    values = np.exp(x - row_max)
    values /= np.sum(values, axis=1, keepdims=True)
    gradient = None # TODO
    return values, gradient

# output layer functions
def sigmoid_binarycrossentropy(Z, Y):
    """Sigmoid ouput layer with binary cross entropy loss for binary classification tasks."""
    Y_pred, _ = sigmoid(Z)
    cost = -np.sum(Y * np.log(Y_pred) + (1.0 - Y) * np.log(1.0 - Y_pred))
    D = Y_pred - Y
    return Y_pred, cost, D

def identity_rmse(Z, Y):
    """Identity output layer with root mean square error (RMSE) for regression tasks."""
    N = Z.shape[0]
    Y_pred = Z
    cost = np.sqrt(np.sum(np.power(Y_pred - Y, 2) / N))
    D = (Y_pred - Y) / N
    return Y_pred, cost, D

def softmax_crossentropy(Z, Y):
    """Softmax output layer with cross entropy loss for multiclass classification tasks."""
    Y_pred, _ = softmax(Z)
    Y_pred += np.ones_like(Y_pred) * 1e-6
    cost = -np.sum(Y * np.log(Y_pred))
    D = Y_pred - Y
    return Y_pred, cost, D

class NeuralNetworkModel:

    def __init__(self):
        self._layer_sizes = []
        self._activation_functions = []
        self._learning_rate = 0.0007
    
    def hidden_layer(self, units, activation):
        """Adds a hidden layer to the network.
        
        Parameters
        ----------
        units : int
            Number of neurons in the layer.
        activation : Callable
            Activation function for the layer."""
        self._layer_sizes.append(units)
        self._activation_functions.append(activation)

    def output_layer(self, units, type):
        """Defines the output layer of the network.
        
        Parameters
        ----------
        units : int
            Number of neurons in the layer. Would be 1 for binary classification and regression, >1 for multiclass classification.
        type : Callable
            Output layer function, comprised of an activation and a cost function."""
        self._layer_sizes.append(units)
        self._output_layer_function = type
    
    def predict(self, X):
        """Predict values for given data.

        Parameters
        ----------
        X : array_like, shape NxD
            The data to predict for.
        
        Returns
        -------
        array_like, shape NxO
            The prediction.
        """
        # forward propagation
        A = X
        for i in range(len(self._layer_sizes) - 1):
            W = self._weights[i]
            b = self._biases[i]
            f = self._activation_functions[i]

            Z = np.dot(A, W) + b
            A, _ = f(Z)
        
        # output layer
        W = self._weights[-1]
        b = self._biases[-1]
        Z = np.dot(A, W) + b
        # TODO refactor cost and output layer function, this contraction is ugly
        Y = np.ones((X.shape[0], self._layer_sizes[-1]))
        Y_pred, _, _ = self._output_layer_function(Z, Y)
        return Y_pred
    
    def fit(self, X, Y, max_iterations=10000, target_cost=1.0):
        """Fit model on data.
        
        Parameters
        ----------
        X : array_like, shape NxD
            The data to train on. N is number of examples, D is number of features.
        Y : array_like, shape NxO
            Labels of the training data. O is one for binary classification and regression problems,
            >1 for multiclass classification problems.
        max_iterations : int, default=10000
            After how many iterations to stop the training.
        target_cost : float, default=1.0
            Training stops if the cost/loss drops below this value.
        
        Returns
        -------
        float
            The final loss value of the training."""
        self._init_model(n_features=X.shape[1])
        cost_history = []

        for i in range(max_iterations):
            cost = self._forward_backward_pass(X, Y)
            cost_history.append(cost)

            if cost <= target_cost:
                break
        
        self._cost_history = cost_history
        return cost_history[-1]
    
    def _init_model(self, n_features):
        self._weights = []
        self._biases = []
        layer_sizes = [n_features] + self._layer_sizes
        for i in range(len(layer_sizes) - 1):
            W = np.random.random((layer_sizes[i], layer_sizes[i+1]))
            b = np.random.random((1, layer_sizes[i+1]))
            self._weights.append(W)
            self._biases.append(b)
    
    def _forward_backward_pass(self, X, Y):
        # TODO refactor and extract the 3 parts (forward, backward, update) into separate functions

        # forward propagation
        As = [X]
        A_grads = [np.ones_like(X)]
        for i in range(len(self._layer_sizes) - 1):
            W = self._weights[i]
            b = self._biases[i]
            f = self._activation_functions[i]

            Z = np.dot(As[-1], W) + b
            A, grad_A = f(Z)
            As.append(A)
            A_grads.append(grad_A)
        
        # output layer
        W = self._weights[-1]
        b = self._biases[-1]
        Z = np.dot(As[-1], W) + b
        _, cost, D = self._output_layer_function(Z, Y)
        
        # backward propagation
        for i in range(1, len(self._layer_sizes)+1):
            A_prev = As[-i]
            A_grad_prev = A_grads[-i]
            W = self._weights[-i]
            b = self._biases[-i]

            grad_W = np.dot(A_prev.T, D)
            grad_b = np.sum(D, axis=0, keepdims=True)
            D = np.dot(D, W.T) * A_grad_prev

            assert grad_W.shape == W.shape
            assert grad_b.shape == b.shape

            # gradient descent step
            W -= self._learning_rate * grad_W
            b -= self._learning_rate * grad_b
        
        return cost