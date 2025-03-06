import numpy
import pandas 
import matplotlib.pyplot as plt
from tqdm import tqdm 

class NN:
    def __init__(self, data, labels):
        # Input data
        self.input_data = data

        self.labels = labels

        # Initialize the layers
        self.input_layer = 784
        self.hidden_layer = 128
        self.output_layer = 10

        # Initialize the weights
        self.W1 = numpy.random.randn(self.hidden_layer, self.input_layer) * numpy.sqrt(2.0/self.input_layer)
        self.W2 = numpy.random.randn(self.output_layer, self.hidden_layer) * numpy.sqrt(2.0/self.hidden_layer)
        self.b1 = numpy.zeros((self.hidden_layer, 1))
        self.b2 = numpy.zeros((self.output_layer, 1))

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        # Initialize the learning rate
        self.learning_rate = 0.01

        # Initialize the number of epochs
        self.epochs = 1500

        self.accuracies = []  # Add this line to store accuracies
        
    def forward_prop(self):
        # Input layer to hidden layer
        self.Z1 = self.W1.dot(self.input_data) + self.b1
        self.A1 = self.ReLU(self.Z1)

        # Hidden layer to output layer
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)        

    def ReLU(self, Z1):
        return numpy.maximum(Z1,0)
    
    def ReLU_derivative(self, dZ1):
        return dZ1 > 0

    def softmax(self, Z):
        # Subtract the maximum value for numerical stability
        Z = Z - numpy.max(Z, axis=0, keepdims=True)
        exp_values = numpy.exp(Z)
        return exp_values / numpy.sum(exp_values, axis=0, keepdims=True)
        
    def one_hot(self, x):
        one_hot = numpy.zeros((x.size, x.max() + 1))
        one_hot[numpy.arange(x.size), x] = 1
        return one_hot.T

    def backward_prop(self):
        one_hot_labels = self.one_hot(self.labels)
        dZ2 = self.A2 - one_hot_labels
        dW2 = 1/m * numpy.dot(dZ2, self.A1.T)
        db2 = 1/m * numpy.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_derivative(self.Z1)
        dW1 = 1/m * numpy.dot(dZ1, self.input_data.T)
        db1 = 1/m * numpy.sum(dZ1)

        return dW1, db1, dW2, db2
        
    def update_params(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2


    def get_accuracy(self, predictions):
        return numpy.sum(predictions == self.labels) / self.labels.size

    def get_predictions(self):
        return numpy.argmax(self.A2, 0)

    def train(self):
        best_accuracy = 0
        patience = 0
        patience_limit = 10


        for epoch in range(self.epochs):
            self.forward_prop()
            dW1, db1, dW2, db2 = self.backward_prop()
            self.update_params(dW1, db1, dW2, db2)
        
            if epoch % 10 == 0:
                predictions = self.get_predictions()
                accuracy = self.get_accuracy(predictions)
                self.accuracies.append(accuracy)
                print(f"Epoch: {epoch} -- Accuracy: {accuracy}")

                # Early stopping
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience = 0
                else:
                    patience += 1
                    
                if patience == patience_limit:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
        
        # Plot accuracy evolution and save figure
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.epochs, 10), self.accuracies, 'b-')
        plt.title('Accuracy Evolution During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig('figures/accuracy_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def test_set(self):
        # Load and preprocess test data
        data = pandas.read_csv('MNIST_CSV/mnist_test.csv')
        data = numpy.array(data)
        data_T = data.T
        
        # Split into labels and pixels
        test_labels = data_T[0]
        test_pixels = data_T[1:] / 255.0  # Normalize pixel values
        
        # Forward pass through the network
        Z1 = self.W1.dot(test_pixels) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        
        # Get predictions
        predictions = numpy.argmax(A2, 0)
        
        # Calculate accuracy
        test_accuracy = numpy.sum(predictions == test_labels) / test_labels.size
        
        print("=" * 50)
        print(f"Test Set Performance:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("=" * 50)
        
        # Plot confusion matrix (optional but informative)
        plt.figure(figsize=(10, 8))
        cm = numpy.zeros((10, 10), dtype=int)
        for i in range(len(test_labels)):
            cm[test_labels[i]][predictions[i]] += 1
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix on Test Set')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def one_try(self):
        while True:
            i = input("Enter the index of the image you want to predict (0-5999) or 'quit' to exit: ")
            if i.lower() == 'quit':
                print("Exiting prediction mode...")
                break
                
            try:
                i = int(i)
                if i < 0 or i >= len(self.labels):
                    print(f"Please enter a valid index between 0 and {len(self.labels)-1}")
                    continue
                    
                pixels = self.input_data.T[i].reshape(-1, 1)  # Reshape to column vector
                label = self.labels[i]
                
                # Forward pass through the network
                Z1 = self.W1.dot(pixels) + self.b1
                A1 = self.ReLU(Z1)
                Z2 = self.W2.dot(A1) + self.b2
                A2 = self.softmax(Z2)
                
                # Get the prediction
                prediction = numpy.argmax(A2)
                print(f"Neural Network Prediction: {prediction}")
                print(f"Actual Label: {label}")
                print(f"Correct!" if prediction == label else "Incorrect!")

                # Display the image
                image = pixels.reshape(28, 28)
                plt.imshow(image, cmap="gray")
                plt.title(f"Label: {label}, Prediction: {prediction}")
                plt.axis("on")
                plt.show()
                
            except ValueError:
                print("Please enter a valid number or 'quit'")

if __name__ == '__main__':
    data = pandas.read_csv('MNIST_CSV/mnist_train.csv')
    data = numpy.array(data)
    m, n = data.shape

    data_T = data.T

    labels = data_T[0] # 5999 labels
    pixels = data_T[1:n] # 5999 images of 784 pixels each (28x28)
    pixels = pixels / 255.0 # Normalize pixel values to be between 0 and 1

    nn = NN(pixels, labels)
    nn.train()
    nn.test_set()  
    nn.one_try()

