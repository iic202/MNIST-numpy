import numpy
import pandas 
import matplotlib.pyplot as plt

class NN:
    def __init__(self):

        # Initialize the layers
        self.input_layer = 784
        self.hidden_layer = 50
        self.output_layer = 10

        # Initialize the weights
        self.W1 = numpy.random.uniform(-0.5, 0.5, (self.hidden_layer, self.input_layer))
        self.W2 = numpy.random.uniform(-0.5, 0.5, (self.output_layer, self.hidden_layer))
        self.b1 = numpy.zeros((self.hidden_layer, 1))
        self.b2 = numpy.zeros((self.output_layer, 1))

        # Initialize the learning rate
        self.learning_rate = 0.1

        # Initialize the number of epochs
        self.epochs = 10
        
    def forward_prop(self, input_data):
        #Z1 = numpy.dot(self.W1, input_data) + self.b1
        Z1 = self.W1.dot(input_data) + self.b1
        A1 = self.ReLU(Z1)

        # Z2 = numpy.dot(self.W2, A1) + self.b2
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)        

    def ReLU(self, Z1):
        return numpy.maximum(Z1,0)
    
    def ReLU_derivative(self, dZ1):
        return dZ1 > 0

    def softmax(self, Z2):
        return numpy.exp(Z2) / numpy.sum(numpy.exp(Z2))
        
    def one_hot(self, x):
        one_hot = numpy.zeros((x.size, x.max() + 1))
        one_hot[numpy.arange(x.size), x] = 1
        return one_hot.T

    def backward_prop(self):
        pass

    def train(self):
        pass

if __name__ == '__main__':
    data = pandas.read_csv('MNIST_CSV/mnist_train.csv')

    data = numpy.array(data)
    m, n = data.shape
    print(f"Data shape: {data.shape}")

    data_T = data.T
    print(f"Data shape: {data_T.shape}")

    labels = data_T[0] # 5999 labels
    pixels = data_T[1:n] # 5999 images of 784 pixels each (28x28)

    print(f"Label shape: {labels.shape}")
    print(f"Pixels shape: {pixels.shape}")

    new_nn = NN()
    new_label = new_nn.one_hot(labels)
    print(f"New label shape: {new_label.shape}")
    print(f"Head of new label: {new_label[:,0:8]}")
    print(f"Head of labels: {labels[0:5]}")



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
    

