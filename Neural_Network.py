import numpy as np

class NNN:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate

        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / (layer_sizes[i]))) #innitialize wights as this, since it is optimal for regression( spefically for Relu)
                                                                                                                     #activations                                                                          
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def compute_loss(self, y, y_hat): #MSE, good metric for my regression models
        return np.mean((y - y_hat) ** 2)

    def compute_accuracy(self, y, y_hat, tolerance=0.1):
        # Avoid division by zero
        y = y.flatten()
        y_hat = y_hat.flatten()
        relative_error = np.abs((y - y_hat) / y)
        correct_predictions = (relative_error <= tolerance).astype(float)
        accuracy = np.mean(correct_predictions) * 100  # Convert to percentage
        return accuracy

    def forward(self, X):
        a_vector = [X]  # Passs input to forward propagation 
        z_vector = []

        for i in range(len(self.weights)):
            z = np.dot(a_vector[-1], self.weights[i]) + self.biases[i]
            z_vector.append(z)
            a = self.relu(z) #Relu Activavtion function
            a_vector.append(a)

        y_hat = a_vector[-1]
        return a_vector, z_vector, y_hat

    def backward(self, a_vector, z_vector, y, y_hat):
        weight_grads = [None] * len(self.weights)
        bias_grads = [None] * len(self.biases)

        delta = (y_hat - y)  # Output layer delta
        for l in range(len(self.weights) - 1, -1, -1):
            weight_grads[l] = np.dot(a_vector[l].T, delta)
            bias_grads[l] = np.sum(delta, axis=0, keepdims=True)
            if l > 0:  # Skip backprop for input layer
                delta = np.dot(delta, self.weights[l].T) * self.relu_derivative(z_vector[l - 1])

        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * weight_grads[l]
            self.biases[l] -= self.learning_rate * bias_grads[l]

    def train(self, X, y, epochs):#since my dataset is small, I am using batch gradient decent for optimizing weights
    #which can provide me with optimal results without caring much for time constarints as my dataset is small
        for epoch in range(epochs):
            a_vector, z_vector, y_hat = self.forward(X)
            
            accuracy = self.compute_accuracy(y, y_hat, 0.1)
            self.backward(a_vector, z_vector, y, y_hat)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")
                
    def test(self, X):
    # Perform a forward pass and return the predictions (y_hat)
     _, _, y_hat = self.forward(X)
     return y_hat

                