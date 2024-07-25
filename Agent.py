import numpy as np
class Agent:
    def __init__(self, lr, gamma, n_actions, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}  #initialize an empty dictionary to store the q values

        #greedy-epsilon
    def choose_actions(self, state):
        if np.random.rand() < self.epsilon:  #select random action if less than epsilon
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q.get((state, a), 0.0) for a in range(self.n_actions)])
            action = np.argmax(actions)   #select action that gives max q value
        return action

    def learn(self, state, action, reward, next_state):
        #select the max q value in the next state
        actions_next = np.array([self.Q.get((next_state, a), 0.0) for a in range(self.n_actions)])  #create np array
        a_max = np.argmax(actions_next)   # find the index of max q value

        # bellman update equation (updating the state, action pair in the q table)
        self.Q[(state, action)] = self.Q.get((state, action), 0.0) + self.lr * (
            reward + self.gamma * actions_next[a_max] - self.Q.get((state, action), 0.0)
        )

        self.decrement_epsilon()

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
    # after model has learned (i.e. q table converged), we can now feed in a state and the model will 
    # give us the optimal action.
    def final_model(self,state):
        actions = np.array([self.Q.get((state,a), 0.0) for a in range(self.n_actions)])
        action = np.argmax(actions)
        if action == 0:
            out = "up"
        elif action == 1:
            out = "down"
        elif action == 2:
            out = "left"
        else:
            out = "right"
        return out

from usercode.Agent import Agent

if __name__ == '__main__':
    env = GridEnv(square_grid_size=10)
    agent = Agent(gamma=0.995,lr = 0.01, n_actions =4,
          eps_dec=0.9, eps_end= 0.01, eps_start=1.0)
    scores = []
    n_eps = 1000  


    for i in range(n_eps):
        done = False
        state = env.reset()
        score = 0
    
        while not done:
            action = agent.choose_actions(state)
            state_, reward, done = env.step(action)
            agent.learn(state, action, reward, state_)
            score += reward
            state = state_
        
 
        scores.append(score)

