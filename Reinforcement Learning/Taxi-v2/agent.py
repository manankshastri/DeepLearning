import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha = 0.4,  gamma = 0.9, epsilon = 0.01):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    
    def select_action(self, state, i_episode):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon =  1.0 / ((i_episode + 10))
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[state])]= 1 - self.epsilon + (self.epsilon / self.nA)
        
        return np.random.choice(np.arange(self.nA), p=policy_s)
   
    def step(self, state, action, reward, next_state, done, i_episode):
        
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """        
               
        
        self.epsilon =  1.0 / ((i_episode + 10))
        policy= np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])]= 1 - self.epsilon + (self.epsilon / self.nA)
             
        self.Q[state][action] += (self.alpha * (reward + (self.gamma * np.dot(self.Q[next_state], policy)) - self.Q[state][action]))