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
        self.hidden_layer = 50
        self.output_layer = 10

        # Initialize the weights
        self.W1 = numpy.random.uniform(-0.5, 0.5, (self.hidden_layer, self.input_layer))
        self.W2 = numpy.random.uniform(-0.5, 0.5, (self.output_layer, self.hidden_layer))
        self.b1 = numpy.zeros((self.hidden_layer, 1))
        self.b2 = numpy.zeros((self.output_layer, 1))

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        # Initialize the learning rate
        self.learning_rate = 0.01

        # Initialize the number of epochs
        self.epochs = 1000

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
        for epoch in range(self.epochs):
            self.forward_prop()
            dW1, db1, dW2, db2 = self.backward_prop()
            self.update_params(dW1, db1, dW2, db2)
            
            # Calculate and store accuracy every 10 epochs
            if epoch % 10 == 0:
                predictions = self.get_predictions()
                accuracy = self.get_accuracy(predictions)
                self.accuracies.append(accuracy)
                print(f"Epoch: {epoch}")
                print(f"Accuracy: {accuracy}")
        
        # Plot accuracy evolution
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.epochs, 10), self.accuracies, 'b-')
        plt.title('Accuracy Evolution During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

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
                    
                # Get the image pixels and reshape for network input
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
    #print(f"Data shape: {data.shape}")

    data_T = data.T
    #print(f"Data shape: {data_T.shape}")

    labels = data_T[0] # 5999 labels
    pixels = data_T[1:n] # 5999 images of 784 pixels each (28x28)
    # Normalize pixel values to be between 0 and 1
    pixels = pixels / 255.0

    #print(f"Label shape: {labels.shape}")
    #print(f"Pixels shape: {pixels.shape}")

    nn = NN(pixels, labels)
    nn.train()  
    nn.one_try()

    # Randomly select an image from the dataset and check if the model can predict the label





    '''
    Plotting a image from the dataset
        i = 0 # This will be the element accesesed
        first_row = data.iloc[i]

        label = first_row.iloc[0]  # First column is the label
        pixels = first_row.iloc[1:].values  # The rest are pixel values

        # Reshape the pixels into a 28x28 image
        image = pixels.reshape(28, 28)

        # Plot the image
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()
    '''


