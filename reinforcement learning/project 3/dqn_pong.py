# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########

        self.network = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )             
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6,
                 beta0=0.4, beta_frames=2000000,
                 eps=1e-6):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta0       = beta0
        self.beta_frames = beta_frames
        self.eps         = eps

        self.buffer      = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.pos         = 0
        self.t           = 1       # times of sampling

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        """Store transition with priority. If error=None use current max."""
        p = (abs(error) + self.eps) if error is not None else self.priorities.max(initial=1.0)
        p = p ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:                       
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n      = len(self.buffer)
        probs  = self.priorities[:n] / self.priorities[:n].sum()

        idxs   = np.random.choice(n, batch_size, p=probs)
        beta   = min(1.0, self.beta0 + (1.0 - self.beta0) * (self.t / self.beta_frames))
        self.t += 1

        weights = (n * probs[idxs]) ** (-beta)
        weights /= weights.max()                       
        weights  = torch.tensor(weights, dtype=torch.float32)

        batch = [self.buffer[i] for i in idxs]
        return idxs, batch, weights

    def update_priorities(self, idxs, errors):
        for i, err in zip(idxs, errors):
            self.priorities[i] = (abs(err) + self.eps) ** self.alpha
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # without PER
        # self.memory = deque(maxlen=args.memory_size)
        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=0.6,        
            beta0=0.4)         
        #
        dimension=self.preprocessor.frame_stack
        self.q_net = DQN(dimension,self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(dimension,self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        ## RMSprop
        self.optimizer = optim.RMSprop(
            self.q_net.parameters(),
            lr=args.lr,          
            alpha=0.95,          
            eps=1e-2,           
            momentum=0.0         
            )

        ## multi-step replay setting
        self.n_step = args.n_step
        self.n_gamma = args.discount_factor ** self.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)  


        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    ## Multi-step-return function
    def add_n_step_transition(self):
        """If n‑step buffer full, push aggregated transition to replay."""
        if (len(self.n_step_buffer)) < self.n_step:
            return
        s0, a0, dum1, dum2, dum3 = self.n_step_buffer[0]
        cumulated_reward, done_f = 0.0, False
        for idx, (dum4, dum5, r, dum6, d) in enumerate(self.n_step_buffer):
            cumulated_reward += (self.gamma ** idx) * r
            if d:
                done_f = True
                break
        dum7, dum8, dum9, state_n, dum10 = self.n_step_buffer[-1]
        ## without PER
        # self.memory.append((s0, a0, cumulated_reward, state_n, done_f))
        max_p = self.memory.priorities.max(initial=1.0)
        self.memory.add((s0, a0, cumulated_reward, state_n, done_f), error=max_p)


    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            
            ## multi-step setting
            self.n_step_buffer.clear()

            ##

            
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                ## reward clipping
                reward=np.sign(reward)
                
                ## store in n-step buffer
                self.n_step_buffer.append((state, action, reward, next_state, done))
                self.add_n_step_transition()
                #self.memory.append((state, action, reward, next_state, done))
                ##

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

                if self.env_count % 200000 == 0:
                    model_path = os.path.join(self.save_dir, f"model_env_count{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)               
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    
                    # drive path
                    drive_path = os.path.join("/content/drive/MyDrive/lab 5/pong_convergence", f"model_env_count{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), drive_path)
                    print(f"Saved model to {model_path}")
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)

                # drive path
                drive_path = os.path.join("/content/drive/MyDrive/lab 5/pong_convergence", f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), drive_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    
                    # drive path
                    drive_path = os.path.join("/content/drive/MyDrive/lab 5/pong_convergence", "best_model.pt")
                    torch.save(self.q_net.state_dict(), drive_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        #state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)
            #state = next_obs

        return total_reward


    def train(self):

        if len(self.memory.buffer) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epislon = self.epsilon_min
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########

        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        ## origin
        #batch = random.sample(self.memory, self.batch_size)
        #states, actions, rewards, next_states, dones = zip(*batch)
        idxs, batch, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        is_weights = is_weights.to(self.device)
       
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########

        # Implement the loss function of DQN and the gradient updates 
        #q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        

        ## origin
        # with torch.no_grad():
        #     next_actions = self.q_net(next_states).argmax(1)  ## picks action by a_net
        #     next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        #     target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # loss = nn.MSELoss()(q_values, target_q_values)

        # self.optimizer.zero_grad()   
        # loss.backward()
        # nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)             
        # self.optimizer.step()
        
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)         
            next_q = self.target_net(next_states).gather(1,next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q

        td_error = target - q_values                      
        loss     = (is_weights * td_error.pow(2)).mean()        

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        #update priorities
        self.memory.update_priorities(idxs, td_error.detach().cpu().numpy())
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-v5-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=5)
    parser.add_argument("--n-step", type=int, default=3)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-pong-v5", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()