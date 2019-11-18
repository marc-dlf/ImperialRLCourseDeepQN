############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import time
from collections import deque

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # DeepQNetwork defining agent actions
        self.dqn = DQN()
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(min_size=50)
        #Reward
        self.total_reward = 0
        #epsilon
        self.epsilon = 1
        self.delta = 0.001

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.epsilon = 1
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        input_tensor = torch.tensor(state.reshape(1,2).astype(np.float32))
        q_values = self.dqn.q_network.forward(input_tensor)
        greedy_action = int(q_values.argmax(1).numpy())

        actions = np.arange(0,4)
        probas = np.ones(4)*(self.epsilon/4)
        probas[greedy_action] = 1-self.epsilon + (self.epsilon/4)
        discrete_action = np.random.choice(actions,p=probas)

        self.epsilon -= self.delta

        action = self._discrete_action_to_continuous(discrete_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add this transition to the replay buffer
        transition_discrete = (self.state, self._continuous_action_to_discrete(self.action), reward, next_state)
        self.replay_buffer.append(transition_discrete)

        if self.replay_buffer.is_full_enough():
            mini_batch = self.replay_buffer.get_minibatch()
            loss = self.dqn.train_q_network(mini_batch)

        self.total_reward+= reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self,state):
        input_tensor = torch.tensor(state.reshape(1,2).astype(np.float32))
        q_values = self.dqn.q_network.forward(input_tensor)
        action = self._discrete_action_to_continuous(int(q_values.argmax(1).numpy()))
        return action

        # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:#Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        elif discrete_action == 2:#Move left
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        else :#Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        return continuous_action

    def _continuous_action_to_discrete(self, continuous_action):
        if (continuous_action == np.array([0.02, 0], dtype=np.float32)).all():  # Move right
            discrete_action = 0
        elif (continuous_action == np.array([0, -0.02], dtype=np.float32)).all():#Move down
            discrete_action = 1
        elif (continuous_action == np.array([-0.02, 0], dtype=np.float32)).all():#Move left
            discrete_action = 2
        elif (continuous_action == np.array([0, 0.02], dtype=np.float32)).all() :#Move up
            discrete_action = 3
        else:
            raise ValueError('not one of actions permited')
        return discrete_action

class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        loss_value = loss.item()
        return loss_value


    def update_target_network(self):
        parameters_q = torch.nn.Module.state_dict(self.q_network)
        torch.nn.Module.load_state_dict(self.target_network,parameters_q)

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self,transition):

        gamma = 0.9
        batch_size = len(transition)
        transition = np.array(transition).reshape(batch_size,4)

        states_batch = np.array([state.reshape(1,2) for state in transition[:,0]])
        next_states_batch = np.array([state.reshape(1,2) for state in transition[:,3]])

        ### extracting batches array from transition

        input_batch =  states_batch.reshape(batch_size,2).astype(np.float32)
        action_batch = transition[:,1].reshape(batch_size,1).astype(np.long)
        reward_batch = transition[:,2].reshape(batch_size,1).astype(np.float32)
        next_states_batch = next_states_batch.reshape(batch_size,2).astype(np.float32)

        ### transforming arrays into tensors

        input_tensor = torch.tensor(input_batch)
        action_tensor = torch.tensor(action_batch)
        reward_tensor = torch.tensor(reward_batch)
        next_states_tensor = torch.tensor(next_states_batch)

        ### Computing the predicted Q value for state s

        q_values = self.q_network.forward(input_tensor)
        q_values = torch.gather(q_values,1,action_tensor)

        ### Computing the actual Q value for this step

        #target_q_values = self.q_network.forward(next_states_tensor)
        target_q_values = self.target_network.forward(next_states_tensor)
        target_q_values = torch.gather(target_q_values,1, target_q_values.argmax(1).view(batch_size,1))

        predicted_sum_future_rewards = reward_tensor.add(target_q_values*gamma)
        loss = torch.nn.MSELoss()(q_values, predicted_sum_future_rewards)

        return loss

    def get_greedy_action(self,state,mode):
        best_reward = -100
        best_action = -100

        if mode==0:
            for i in range(4):
                input_state_action = np.append(state,i).reshape(1,3).astype(np.float32)
                input_tensor = torch.tensor(input_state_action)
                predicted_reward = self.q_network.forward(input_tensor)

                if predicted_reward>best_reward:
                    best_reward = predicted_reward
                    best_action = i


        elif mode==1:
            input_tensor = torch.tensor(state.reshape(1,2).astype(np.float32))
            q_values = self.q_network.forward(input_tensor)
            best_action = int(q_values.argmax(1).numpy())

        else:
            raise ValueError('mode must be either 0 or 1')

        return best_action

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

class ReplayBuffer(deque):
    #Find moment where agent has difficulties and focus on them : for example, best reward since long time
    def __init__(self,min_size):
        super().__init__([],10**6)
        self.min_size = min_size

    def is_full_enough(self):
        return len(self)>=self.min_size

    def get_minibatch(self):
        idx = np.random.choice(np.arange(0,len(self)),self.min_size,replace=False)
        try:
            minibatch = []
            for i in range(len(idx)):
                minibatch += [self[idx[i]]]
        except:
            raise ValueError("Replay Buffer is not full enough")

        return np.array(minibatch)
