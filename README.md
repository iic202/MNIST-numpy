# MNIST Neural Network in NumPy  

This project implements a simple **fully connected neural network** to classify handwritten digits from the **MNIST dataset** using only **NumPy** (no TensorFlow/Keras).  


## Overview  

The **MNIST dataset** contains **70,000 grayscale images (28x28 pixels)** of handwritten digits (0-9).  
This project implements a **basic 2-layer neural network**:  

1. **Input Layer** – 784 neurons (28×28 pixels)  
2. **Hidden Layer** – 128 neurons (ReLU activation)  
3. **Output Layer** – 10 neurons (Softmax activation)  
4. **Iteration** - 1250 iterations


## Accuracy

As you can see in the figure below, the accuracy with 128 neurons in the hidden layer tends to get close to 90% accuracy, with 1000 iterations we get close to that limit. We could increase iterations and neurons in the hidden layer and this would increase the accuracy by about 3-5%. For example with 200 neurons and 2000 iterations this neural network gets about 91%-93% accuracy. While 128 neurons and 1000 iterations gets about 87%-89%. 

![Accuracy evolving over iterations](figures/accuracy_evolution.png)

### Acuracy on test set

![Confusion matrix on test set](figures/confusion_matrix.png)

