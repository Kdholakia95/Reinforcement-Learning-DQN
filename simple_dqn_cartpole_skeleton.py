import gym
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop

def Q_Net(input_size, hidden1_size, hidden2_size, output_size):
	X_input = Input(input_size)
	X = Dense(hidden1_size, input_shape=input_size, activation="relu", kernel_initializer='he_uniform')(X_input)
	X = Dense(hidden2_size, activation="relu", kernel_initializer='he_uniform')(X)
	X = Dense(output_size, activation="linear", kernel_initializer='he_uniform')(X)

	model = Model(inputs = X_input, outputs = X)
	model.compile(loss="mse", optimizer=RMSprop(lr=0.0002, rho=0.95, epsilon=0.01), metrics=["accuracy"])

	return model

class DQN:
	
	REPLAY_MEMORY_SIZE = 10000 			# number of tuples in experience replay  
	EPSILON = 1 					# epsilon of epsilon-greedy exploation
	EPSILON_DECAY = 0.999 				# exponential decay multiplier for epsilon
	HIDDEN1_SIZE = 128 				# size of hidden layer 1
	HIDDEN2_SIZE = 128 				# size of hidden layer 2
	EPISODES_NUM = 300 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 200 				# maximum number of steps in an episode 
	LEARNING_RATE = 0.0001 				# learning rate and other parameters for SGD/RMSProp/Adam
	MINIBATCH_SIZE = 100 				# size of minibatch sampled from the experience replay
	DISCOUNT_FACTOR = 0.95 				# MDP's gamma
	TARGET_UPDATE_FREQ = 100			# number of steps (not episodes) after which to update the target networks 
	LOG_DIR = './logs' 				# directory wherein logging takes place	

	# Create and initialize the environment
	def __init__(self, env):
		self.env = gym.make(env)
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n			# In case of cartpole, 2 actions (right/left)
		self.use_target_network = True
		self.use_replay_memory = True
							# Training Net
		self.training_model = Q_Net(input_size=(self.input_size,), hidden1_size = self.HIDDEN1_SIZE, hidden2_size = self.HIDDEN2_SIZE, output_size = self.output_size)
							# Target Net
		self.target_model = Q_Net(input_size=(self.input_size,), hidden1_size = self.HIDDEN1_SIZE, hidden2_size = self.HIDDEN2_SIZE, output_size = self.output_size)
	
	def train(self, episodes_num=EPISODES_NUM):
		
						# Initialize summary for TensorBoard 						
		self.summary_writer = tf.summary.create_file_writer(self.LOG_DIR)

		# Alternatively, you could use animated real-time plots from matplotlib 
		# (https://stackoverflow.com/a/24228275/3284912)		
		
		############################################################
		# Initialize other variables (like the replay memory)
		############################################################
		replay_buffer = []
		buffer_top = 0
		total_steps = 0
		avg_reward = np.zeros(100)
		avg_reward_top = 0
		optimal_Q_count = 0		

		############################################################
		# Main training loop
		# 
		# In each episode, 
		#	pick the action for the given state, 
		#	perform a 'step' in the environment to get the reward and next state,
		#	update the replay buffer,
		#	sample a random minibatch from the replay buffer,
		# 	perform Q-learning,
		#	update the target network, if required.
		#
		#
		#
		# You'll need to write code in various places in the following skeleton
		#
		############################################################

		for episode in range(episodes_num):
		  
			state = self.env.reset()
			state = np.reshape(state, [1, self.input_size])
			episode_length = 0
			loss = 0

			############################################################
			# Episode-specific initializations go here.
			############################################################
			#
			# Your code here
			#
			############################################################

			while True:

				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################
				if np.random.uniform(0, 1) < self.EPSILON:      # Exploration
					action = (int)(np.random.randint(0, 2, 1))
				else:                                           # Exploitation using training net to predict best action
					action = np.argmax(self.training_model.predict(state))
					
						# Performing epsilon-decay with capped minimum epsilon
				self.EPSILON = max(0.001, self.EPSILON * self.EPSILON_DECAY) 
				############################################################
				# Step in the environment. Something like: 
				# next_state, reward, done, _ = self.env.step(action)
				############################################################
				next_state, reward, done, _ = self.env.step(action)
				episode_length += 1
				total_steps += 1
				
				next_state = np.reshape(next_state, [1, self.input_size])				
					
				current_experience = [state, action, reward, next_state, done]

				############################################################
				# Update the (limited) replay buffer. 
				#
				# Note : when the replay buffer is full, you'll need to 
				# remove an entry to accommodate a new one.
				############################################################
								# if replay_buffer is not full
				if len(replay_buffer) < self.REPLAY_MEMORY_SIZE:
					replay_buffer.append(current_experience)                                        
								# if replay_buffer is full
				else:
					replay_buffer[buffer_top] = current_experience

				buffer_top += 1
				buffer_top %= self.REPLAY_MEMORY_SIZE							
				
				############################################################
				# Sample a random minibatch and perform Q-learning (fetch max Q at s') 
				#
				# Remember, the target (r + gamma * max Q) is computed    
				# with the help of the target network.
				# Compute this target and pass it to the network for computing 
				# and minimizing the loss with the current estimates
				#
				############################################################

				if self.use_replay_memory and len(replay_buffer) >= 1000:#self.MINIBATCH_SIZE:
					batch = random.sample(replay_buffer, self.MINIBATCH_SIZE)
					
					state_list = np.zeros((self.MINIBATCH_SIZE, self.input_size))
					action_list = np.zeros(self.MINIBATCH_SIZE, dtype=int)
					next_state_list = np.zeros((self.MINIBATCH_SIZE, self.input_size))
					reward_list = np.zeros(self.MINIBATCH_SIZE)					
					done_list = []

					k = 0
					for experience in batch:						
						state_list[k] = experience[0]
						action_list[k] = experience[1]
						reward_list[k] = experience[2]
						next_state_list[k] = experience[3]
						done_list.append(experience[4])						
						k += 1						
					
					state_list = np.array(state_list, )
					action_list = np.array(action_list)
					next_state_list = np.array(next_state_list)
					reward_list = np.array(reward_list)
					done_list = np.array(done_list)
										
					if not self.use_target_network:
						expected_Q = self.training_model.predict(state_list)
						next_state_expected_Q = self.training_model.predict(next_state_list)
					else:
						expected_Q = self.target_model.predict(state_list)
						next_state_expected_Q = self.target_model.predict(next_state_list)
					
					for j in range(self.MINIBATCH_SIZE):
						if not done_list[j]:
							expected_Q[j][action_list[j]] = reward_list[j] + self.DISCOUNT_FACTOR * np.amax(next_state_expected_Q[j])
						else:
							expected_Q[j][action_list[j]] = reward_list[j]

					predicted_Q = self.training_model.predict(state_list)
					loss = tf.math.reduce_mean(tf.square(expected_Q - predicted_Q)) / self.MINIBATCH_SIZE
					self.training_model.fit(state_list, expected_Q, batch_size=self.MINIBATCH_SIZE, verbose=0)
					
				elif not self.use_replay_memory:
					
					if not self.use_target_network:
						expected_Q = self.training_model.predict(state)
						next_state_expected_Q = self.training_model.predict(next_state)
					else:
						expected_Q = self.target_model.predict(state)
						next_state_expected_Q = self.target_model.predict(next_state)
						
					if not done:
						expected_Q[0][action] = reward + self.DISCOUNT_FACTOR * np.amax(next_state_expected_Q)
					else:
						expected_Q[0][action] = reward

					predicted_Q = self.training_model.predict(state)
					loss = tf.math.reduce_mean(tf.square(expected_Q - predicted_Q)) / self.MINIBATCH_SIZE
					self.training_model.fit(state, expected_Q, batch_size=1, verbose=0)

				state = next_state
				
				############################################################
				# Update target weights. 
				#
				# Something along the lines of:
				# if total_steps % self.TARGET_UPDATE_FREQ == 0:
				# 	target_weights = self.session.run(self.weights)
				############################################################
				
				if self.use_target_network and total_steps % self.TARGET_UPDATE_FREQ == 0:
					self.target_model.set_weights(self.training_model.get_weights())
															
				############################################################
				# Break out of the loop if the episode ends
				#
				# Something like:
				# if done or (episode_length == self.MAX_STEPS):
				# 	break
				#
				############################################################
								      
				if done or (episode_length == self.MAX_STEPS):                                        
					break

			############################################################
			# Logging. 
			#
			# Very important. This is what gives an idea of how good the current
			# experiment is, and if one should terminate and re-run with new parameters
			# The earlier you learn how to read and visualize experiment logs quickly,
			# the faster you'll be able to prototype and learn.
			#
			# Use any debugging information you think you need.
			# For instance :
			  
			avg_reward[avg_reward_top] = episode_length
			avg_reward_top += 1
			avg_reward_top %= 100
			avg_reward[avg_reward_top] = 0
			avg_reward_100 = np.mean(avg_reward)
			
			print("Training: Ep= %d, Rwd= %d, Time= %d, Avg-Rwd= %.3f, epsln= %.3f, Loss= %.3f" % (episode, episode_length, total_steps, avg_reward_100, self.EPSILON, loss))
			with self.summary_writer.as_default():
				tf.summary.scalar('Episode reward vs Episodes', episode_length, step=episode)
				tf.summary.scalar('Average reward over last 100 episodes vs Episodes', avg_reward_100, step=episode)
				tf.summary.scalar('Loss vs Episodes', loss, step=episode)

			if episode_length == self.MAX_STEPS:
				optimal_Q_count += 1
			else:
				optimal_Q_count = 0
						# Reached max_steps atleast 10 times consecutively and minimized loss		
			if episode > 250 and optimal_Q_count >= 20 and loss <= 0.01:
				break
							
		return episode, avg_reward, avg_reward_top


	# Simple function to visually 'test' a policy
	def playPolicy(self):
		
		done = False
		steps = 0
		state = self.env.reset()
		state = np.reshape(state, [1, self.input_size])
		
		# we assume the CartPole task to be solved if the pole remains upright for 100 steps
		while not done and steps < 200: 	
			#self.env.render()				
			q_vals = self.training_model.predict(state)
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			state = np.reshape(state, [1, self.input_size])
			steps += 1
		
		return steps


if __name__ == '__main__':

	# Create and initialize the model
	dqn = DQN('CartPole-v0')
	dqn.use_target_network = False
	dqn.use_replay_memory = False

	print("\nStarting training...\n")
	last_episode, avg_reward, avg_reward_top = dqn.train()
	
	print("\nFinished training...\nCheck out some demonstrations\n")	
	# Visualize the learned behaviour for a few episodes
	results = []
	#for i in range(100):
	while last_episode < 400:
		episode_length = dqn.playPolicy()
		print("Test steps = ", episode_length)
		results.append(episode_length)
		last_episode += 1
		episode = last_episode #+ i		
		avg_reward_top += 1
		avg_reward_top %= 100
		avg_reward[avg_reward_top] = episode_length
		avg_reward_100 = np.mean(avg_reward)
		with dqn.summary_writer.as_default():
			tf.summary.scalar('Episode reward vs Episodes', episode_length, step=episode)
			tf.summary.scalar('Average reward over last 100 episodes vs Episodes', avg_reward_100, step=episode)
	print("Mean steps over 100 episodes = ", sum(results) / len(results))	
	
	print("\nFinished.")
	print("\nCiao, and hasta la vista...\n")
