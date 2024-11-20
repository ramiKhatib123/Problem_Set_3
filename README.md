## Neural Network Model for Predicting Car's Fuel Efficiency 

## **Overview**
This project implements a **custom neural network** from scratch in Python using NumPy. The model is designed for **regression tasks** and is trained on the `auto-mpg` dataset sourced from **Kaggle**. The dataset predicts the fuel efficiency (miles per gallon) of automobiles based on various features like engine displacement, horsepower, weight, etc.

The code includes all essential components of a neural network:
- **Forward Propagation**
- **Backpropagation**
- **Batch Gradient Descent**

Additionally, the implementation calculates **accuracy** based on a tolerance level (e.g., 10%) for regression predictions.

---

## **Dataset**
The dataset used is `auto-mpg`, which was downloaded from Kaggle. This dataset contains the following features:
- **Features**:
  - `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model year`, and `origin`.
- **Target**:
  - `mpg` (miles per gallon) is the target variable we aim to predict.
- **Preprocessing**:
  - The `horsepower` column contains missing values, which were replaced with `NaN` and then handled by dropping rows with missing data.
  - Numerical features were normalized using `StandardScaler`, and categorical features (`origin`) were one-hot encoded (to avoid hvaing the neural network thinking there are any sequential order in the origin encoded values).

---

## **Model Architecture**
The neural network consists of:
1. **Input Layer**:
   - Takes the preprocessed features as input (after normalization and one-hot encoding).

2. **Hidden Layers**:
   - **3 fully connected hidden layers** with the following structure:
     - **Layer 1:** 8 neurons
     - **Layer 2:** 4 neurons
     - **Layer 3:** 2 neurons
   - Activation function: **ReLU** (Rectified Linear Unit) for all hidden layers.

3. **Output Layer**:
   - 1 neuron for predicting the `mpg` (continuous value).
   - Activation function: **Linear (no activation)**.

4. **Loss Function**:
   - Mean Squared Error (MSE) is used as the loss function for optimization.

---

## **Features of the Code**
### **1. Custom Implementation**
This neural network is implemented entirely from scratch using NumPy, without the use of external libraries like TensorFlow or PyTorch. This provides a deeper understanding of the working principles of neural networks.

### **2. Training Process**
The `train` method trains the model using **full-batch gradient descent**, where all data points are used to calculate gradients at each epoch. This approach is feasible because the dataset is small.

### **3. Accuracy Metric**
The accuracy is computed based on a **tolerance level** (e.g., predictions within 10% of the true value are considered accurate). This provides a more interpretable metric for regression tasks compared to the loss.

---

## **How to Use the Code**
### **1. Preprocessing**
The dataset is preprocessed in `Build_data.py`, which includes:
- Handling missing data in `horsepower`.
- Normalizing numerical features.
- One-hot encoding categorical features.

### **2. Neural Network Training**
The neural network is implemented in `Neural_Network.py`. You can configure the following:
- Number of hidden layers.
- Learning rate.
- Number of epochs.

To train the model:
```python
# Train the model
nn.train(X_train, y_train.values.reshape(-1, 1), epochs=1000)
To train the model:

# Test the model
y_pred = nn.test(X_test)
```

### **3. Accuracy**
Evaluate model's performance using the compute_accuracy function:

```python

accuracy = nn.compute_accuracy(y_test.reshape(-1, 1), y_pred, tolerance=0.1)
```
## Conclusion
This project demonstrates the fundamental building blocks of a neural network and provides an end-to-end workflow for training a custom model on a regression dataset. While simple, the model achieves interpretable results and highlights the challenges of implementing machine learning models from scratch.

Feel free to adapt this project further or extend it with additional features like:

- Support for mini-batch gradient descent.
- Regularization techniques.
- Testing with larger datasets.
