import numpy as np
import torch


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# Main entry point
if __name__ == "__main__":

    # Create some input data with 3 dimensions
    input_data = np.random.uniform(0, 1, [100, 3]).astype(np.float32)
    # Create some label data with 2 dimensions
    label_data = np.zeros([100, 2], dtype=np.float32)
    # Create a function which maps the input data to the labels. This is the function which the neural network will try to predict.
    for i in range(100):
        label_data[i, 0] = 1 + input_data[i, 0] * input_data[i, 0] + input_data[i, 1] * input_data[i, 2]
        label_data[i, 1] = input_data[i, 0] * input_data[i, 1] - input_data[i, 2] * 3

    # Create the neural network
    network = Network(input_dimension=3, output_dimension=2)
    # Create the optimiser
    optimiser = torch.optim.Adam(network.parameters(), lr=0.1)

    # Loop over training iterations For
    for training_iteration in range(1000):
        # Set all the gradients stored in the optimiser to zero.
        optimiser.zero_grad()
        # Sample a batch of size 5 from the training data
        # NOTE: when just training on a single example on each iteration), the NumPy array (and Torch tensor) still needs to have two dimensions: the batch dimension, and the data dimension. And in this case, the batch dimension would be 1, instead of 5. This can be done by using the torch.unsqueeze() function.
        batch_indices = np.random.choice(range(100), 5)
        batch_inputs = input_data[batch_indices]
        #print(batch_inputs.shape)
        batch_labels = label_data[batch_indices]
        # Convert the NumPy array into a Torch tensor
        batch_input_tensor = torch.tensor(batch_inputs)
        batch_labels_tensor = torch.tensor(batch_labels)
        # Do a forward pass of the network using the inputs batch
        # NOTE: when training a Q-network, you will need to find the prediction for a particular action. This can be done using the "torch.gather()" function.
        network_prediction = network.forward(batch_input_tensor)
        # Compute the loss based on the label's batch
        loss = torch.nn.MSELoss()(network_prediction, batch_labels_tensor)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the network parameters.
        loss.backward()
        # Take one gradient step to update the network
        optimiser.step()
        # Get the loss as a scalar value
        loss_value = loss.item()
        # Print out the loss
        print('loss = ' + str(loss_value))
