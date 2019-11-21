import numpy as np
import torch
import time
from collections import deque

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 600
        self.episode_length_greedy = 100
        self.num_episode_completed = 0
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        self.num_steps_taken_ep = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # DeepQNetwork defining agent actions
        self.dqn = DQN(gamma = 0.9)
        # Reward
        self.total_reward = 0
        # Epsilon
        self.epsilon = 1
        # Epsilin decay coefficient
        self.delta = 0.001
        ## The point the most on the right the agent reached (episode/ever)
        self.rightmost_ep = -100
        self.rightmost_ever = -100
        ## The last time the reward was positive
        self.last_time_agent_got_more_right = 0
        ## True if the agent just got stuck by a wall on top/bottom
        self.stuck = False
        ## Last distance to goal
        self.last_distance_to_goal = 1
        self.goal_reached_last_ep = False
        ##Direction in tunnel mode (when the agent has not got a positive reward
        ##for a long time, we can suppose it is stuck in a long tunnel when
        ##it has to choose a direction up/down and stick to it until getting
        ##unstuck)
        self.direction_up = None
        #Flag for greedy mode
        self.greedy_mode = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):

        if (not self.greedy_mode):
            # Episode is finished if the agent has taken the maximum number of steps
            # or if it reached the goal
            if ((self.num_steps_taken_ep % self.episode_length == 0) \
                and (self.num_steps_taken_ep!= 0)) or (self.has_reached_goal()):

                self.num_episode_completed+=1

                ## Decreasing the starting value of epsilon after each episode
                if 1-(self.num_episode_completed/80)>0.1:
                    self.epsilon = 1-self.num_episode_completed/80
                else:
                    self.epsilon = 0.1

                ## If the goal was reached we want to reduce the length of an
                ## episode
                if self.has_reached_goal():
                    self.goal_reached_last_ep = True
                    if (self.episode_length-1)>=150:
                        self.episode_length -=1
                else:
                    self.goal_reached_last_ep = False

                ## Starting to test the greedy policy after 10 episodes
                if self.num_episode_completed>=10:
                    self.greedy_mode = not self.greedy_mode

                ## Reseting variables concerning only the episode
                self.rightmost_ep = 0
                self.num_steps_taken_ep = 0

                return True
            else:
                return False

        else:
            if ((self.num_steps_taken_ep % self.episode_length_greedy == 0) \
                and (self.num_steps_taken_ep!= 0)) or (self.has_reached_goal()):
                self.greedy_mode = not self.greedy_mode

                ## stops the training if greedy policy reaches goal in 100 steps
                if self.has_reached_goal():
                    self.dqn.optimiser = torch.optim.Adam(self.dqn.q_network.parameters(), lr=0.)
                print(self.last_distance_to_goal)
                return True
            else:
                False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        greedy_action = self.get_greedy_action(state)

        if not self.greedy_mode:
            # If the agent made no progress for 200 steps --> tunnel mode
            if (self.last_time_agent_got_more_right>=200):
                if self.direction_up is None:
                    # Find maximum q-value between up and down at this position
                    input_tensor = torch.tensor(state.reshape(1,2).astype(np.float32))
                    q_values = self.dqn.q_network.forward(input_tensor).detach()
                    best_action = np.argmax(q_values.reshape(8)[[1,3]])
                    self.direction_up = best_action
                # Inverting the direction if stuck
                if self.stuck == True:
                    self.direction_up = not self.direction_up
                # Here choose only between up and right
                if self.direction_up:
                    discrete_action = np.random.choice([0,3],p=[0.5,0.5])
                # Here choose only between down and right
                else:
                    discrete_action = np.random.choice([0,1],p=[0.5,0.5])
            else:
                self.direction_up=None
                # Choice of action with epsilon greedy policy
                actions = np.arange(0,8)
                probas = np.ones(8)*(self.epsilon/8)
                discrete_greedy = self._continuous_action_to_discrete(greedy_action)
                probas[discrete_greedy] = 1-self.epsilon + (self.epsilon/8)
                discrete_action = np.random.choice(actions,p=probas)
                # Decay of epsilon
                if (self.epsilon-self.delta)>=0.1:
                    self.epsilon -= self.delta
            action = self._discrete_action_to_continuous(discrete_action)

        else:
            action = greedy_action

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        self.num_steps_taken_ep += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action


    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        reward = self.compute_reward(next_state)
        self.last_distance_to_goal = distance_to_goal
        # Update the values where we know if the agent made progress recently
        self.update_state_of_progression()
        self.set_stuck_variable(self.action,self.state,next_state)

        ## Update target network every 100 steps
        if (self.num_steps_taken % 100 == 0):
            self.dqn.update_target_network()

        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add this transition to the replay buffer (state,action,reward,next_state,transition_index,initial wieght)
        discretized_action = self._continuous_action_to_discrete(self.action)
        transition_discrete = (self.state, discretized_action,reward, next_state)

        self.dqn.replay_buffer.add_weight()
        self.dqn.replay_buffer.append(transition_discrete)

        if self.dqn.replay_buffer.is_full_enough():
            mini_batch = self.dqn.replay_buffer.get_minibatch(alpha=0.3)
            # Don't train in greedy_mode
            if not self.greedy_mode:
                loss = self.dqn.train_q_network(mini_batch)

        self.total_reward+= reward


    def compute_reward(self,next_state):
        # The reward is the addition of several bonuses and maluses with a fixed
        # negative initial value (corresponding to the cost of an action)
        reward = -1/10
        # Malus if the agent chooses an action which leads to no movement
        if (self.state == next_state).all():
            reward -= 0.9
        # If the action resulted in a significant movement, the reward
        # corresponds to the number of steps on the right the agent made
        if np.linalg.norm(self.state - next_state)<np.linalg.norm(self.action)/10:
            reward -= 0.1
        else:
            reward += (next_state[0] - self.state[0])*50
        # Bonus if the agent reaches the goal
        if (self.has_reached_goal()):
            reward += 1
        return reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self,state):
        input_tensor = torch.tensor(state.reshape(1,2).astype(np.float32))
        q_values = self.dqn.q_network.forward(input_tensor).detach()
        action = self._discrete_action_to_continuous(int(q_values.argmax(1).numpy()))
        return action

        # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:  # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:#Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        elif discrete_action == 2 :#Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 3: #diago up
            continuous_action = np.array([0.01, 0.01], dtype=np.float32)
        elif discrete_action == 4: #diago down
            continuous_action = np.array([0.01, -0.01], dtype=np.float32)
        elif discrete_action == 5: #forward little
            continuous_action = np.array([0.01, 0.0], dtype=np.float32)
        elif discrete_action == 6: #up little
            continuous_action = np.array([0.00, 0.01], dtype=np.float32)
        elif discrete_action == 7: #down little
            continuous_action = np.array([0.0, -0.01], dtype=np.float32)
        return continuous_action

    def _continuous_action_to_discrete(self, continuous_action):
        if (continuous_action == np.array([0.02, 0], dtype=np.float32)).all():  # Move right
            discrete_action = 0
        elif (continuous_action == np.array([0, -0.02], dtype=np.float32)).all():#Move down
            discrete_action = 1
        elif (continuous_action == np.array([0, 0.02], dtype=np.float32)).all() :#Move up
            discrete_action = 2
        elif (continuous_action == np.array([0.01, 0.01], dtype=np.float32)).all(): #diago up
            discrete_action = 3
        elif (continuous_action == np.array([0.01, -0.01], dtype=np.float32)).all(): #diago down
            discrete_action = 4
        elif (continuous_action == np.array([0.01, 0.0], dtype=np.float32)).all(): #forward little
            discrete_action = 5
        elif (continuous_action == np.array([0.00, 0.01], dtype=np.float32)).all(): #up little
            discrete_action = 6
        elif (continuous_action == np.array([0.0, -0.01], dtype=np.float32)).all(): #down little
            discrete_action = 7
        else:
            raise ValueError('not one of actions permited')
        return discrete_action

    ## Is the agent stuck by a wall on top/bottom
    def set_stuck_variable(self,action,state,next_state):
        if (self._continuous_action_to_discrete(action)==1 or \
            self._continuous_action_to_discrete(action)==3) and \
        (state==next_state).all():
            self.stuck = True
        else:
            self.stuck = False

    ## Has the agent reached the goal
    def has_reached_goal(self):
        return self.last_distance_to_goal<=0.03

    ## See if the agent made any progress recently
    def update_state_of_progression(self):
        if self.rightmost_ep<self.state[0]:
            self.rightmost_ep = self.state[0]
            self.last_time_agent_got_more_right = 0

            if self.rightmost_ever<self.state[0]:
                self.rightmost_ever = self.state[0]
        else:
            self.last_time_agent_got_more_right+= 1

class DQN:

    # The class initialisation function.
    def __init__(self,gamma):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=8)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.target_network = Network(input_dimension=2, output_dimension=8)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(min_size=50,max_size=10**6)

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

        batch_size = len(transition)
        transition = np.array(transition).reshape(batch_size,5)
        transition_idx = transition[:,4].reshape(batch_size).astype(np.int64)

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

        ### Computing the target Q values
        target_q_values = self.target_network.forward(next_states_tensor)
        target_q_values = torch.gather(target_q_values,1, target_q_values.argmax(1).view(batch_size,1))

        predicted_sum_future_rewards = reward_tensor.add(target_q_values*self.gamma)

        ### Update weights
        weights_updated = predicted_sum_future_rewards.detach().numpy().reshape(batch_size) - q_values.detach().numpy().reshape(batch_size)
        self.replay_buffer.weights[transition_idx] = np.abs(weights_updated) + 0.0001

        loss = torch.nn.MSELoss()(q_values, predicted_sum_future_rewards)

        return loss

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=300)
        self.layer_2 = torch.nn.Linear(in_features=300, out_features=300)
        self.output_layer = torch.nn.Linear(in_features=300, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class ReplayBuffer(deque):
    #Find moment where agent has difficulties and focus on them : for example, best reward since long time
    def __init__(self,min_size,max_size):
        super().__init__([],max_size)
        self.min_size = min_size
        self.weights = np.array([])
        self.max_size = max_size

    # Add a new transition and its weight to the buffer
    def add_weight(self):
        if len(self.weights)==0:
            self.weights = np.append(self.weights,1)
        else:
            if len(self)==self.max_size:
                self.weights = np.append(self.weights[1:],np.max(self.weights))
            else:
                self.weights = np.append(self.weights,np.max(self.weights))

    def normalized_weights(self,alpha=0):
        return np.power(self.weights,alpha)/np.sum(np.power(self.weights,alpha))

    def is_full_enough(self):
        return len(self)>=self.min_size

    def discard_weights(self,before):
        self.weights[:before] = 0

    def get_minibatch(self,alpha=0):
        idx = np.random.choice(np.arange(0,len(self)),self.min_size,p=self.normalized_weights(alpha),replace=False)
        try:
            minibatch = []
            for i in range(len(idx)):
                minibatch += [self[idx[i]]+(idx[i],)]
        except:
            raise ValueError("Replay Buffer is not full enough")

        return np.array(minibatch)
