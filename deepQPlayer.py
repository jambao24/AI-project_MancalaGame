import os
import gym
import random
import logging
from collections import namedtuple, deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from mancala import MancalaBoard, Player
from randomPlayer import RandomPlayer

n_steps = 500_000
batch_size = 128
gamma = 0.99
eps_start = 1.0
eps_end = 0.1
eps_steps = 200_000
target_update = 10

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MancalaTrainer(gym.Env):
  reward_range = (-np.inf, np.inf)

  def __init__(self) -> None:
    self.board = MancalaBoard()
    self.isPlayer1Turn = True
    self.summary = {
        "total games": 0,
        "ties": 0,
        "player 1 wins": 0,
        "player 2 wins": 0,
    }

  def reset(self):
    self.board = MancalaBoard()
    return self.board.board

  def step(self, i):
    goAgain = False
    if self.board.board[i] == 0:
      reward = -10
    else:
      goAgain = self.board.playPit(i)
      reward = 10 if self.board.isPlayer1Winning() else 0
    if self.board.isGameOver():
      self.summary["total games"] += 1
      if self.board.isGameTie(): self.summary["ties"] += 1
      elif self.board.isPlayer1Winning(): self.summary["player 1 wins"] += 1
      else: self.summary["player 2 wins"] += 1
    return (self.board.board, reward, self.board.isGameOver(), goAgain)

  def render(self, mode: str = "human"):
    print("{}|{}|{}|{}|{}|{}\n{}|\t\t|{}\n{}|{}|{}|{}|{}|{}".format(
        self.board.board[7:13].reverse() + self.board.board[13] + self.board.board[6] + self.board.board[:6]
    ))

class DeepQPlayer(Player):

  def __init__(self, isPlayer1) -> None:
    super().__init__(isPlayer1)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(os.path.exists("models/dqnModel.pt")):
      self.model = load_model("models/dqnModel.pt", self.device)
    else:
      self.model = None

  def getNextMove(self, boardState: np.array) -> int:
    action, _ = select_model_action(self.device, self.model, torch.tensor([boardState], dtype=torch.float).to(device), 0)
    return action.item()

  def train(self):
    logging.info("Beginning training on: {}".format(device))
    old_summary = {
        "total games": 0,
        "ties": 0,
        "player 1 wins": 0,
        "player 2 wins": 0,
    }
    policy = Policy(n_inputs=14, n_outputs=6).to(device)
    target = Policy(n_inputs=14, n_outputs=6).to(device)
    player2 = RandomPlayer(False)
    target.load_state_dict(policy.state_dict())
    target.eval()
    env = MancalaTrainer()
    state = torch.tensor([env.reset()], dtype=torch.float).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    player1GoAgain = False
    player2GoAgain = False
    for step in range(n_steps):
      t = np.clip(step / eps_steps, 0, 1)
      eps = (1 - t) * eps_start + t * eps_end

      action, _ = select_model_action(device, policy, state, eps)
      next_state, reward, done, player1GoAgain = env.step(action.item())

      # player 2 goes
      if not done or player1GoAgain:
          while(True):
            next_state, _, done, player2GoAgain = env.step(player2.getNextMove(next_state))
            if done or not player2GoAgain: break
          next_state = torch.tensor([next_state], dtype=torch.float).to(device)
      if done:
          next_state = None

      memory.push(state, action, next_state, torch.tensor([reward], device=device))

      state = next_state
      optimize_model(
          device=device,
          optimizer=optimizer,
          policy=policy,
          target=target,
          memory=memory,
          batch_size=batch_size,
          gamma=gamma,
      )
      if done:
          state = torch.tensor([env.reset()], dtype=torch.float).to(device)
      if step % target_update == 0:
          target.load_state_dict(policy.state_dict())
      if step % 5000 == 0:
        delta_summary = {k: env.summary[k] - old_summary[k] for k in env.summary}
        old_summary = {k: env.summary[k] for k in env.summary}
        logging.info("{} : {}".format(step, delta_summary))
    torch.save(policy.state_dict(), "models/dqnModel.pt")


class Policy(nn.Module):

    def __init__(self, n_inputs=3*12, n_outputs=6):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
    def act(self, state):
        with torch.no_grad():
            return self.forward(state).max(1)[1].view(1, 1)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(50_000)

def optimize_model(
    device: torch.device,
    optimizer: optim.Optimizer,
    policy: Policy,
    target: Policy,
    memory: ReplayMemory,
    batch_size: int,
    gamma: float,
):
    """Model optimization step, copied verbatim from the Torch DQN tutorial.
    
    Arguments:
        device {torch.device} -- Device
        optimizer {torch.optim.Optimizer} -- Optimizer
        policy {Policy} -- Policy net
        target {Policy} -- Target net
        memory {ReplayMemory} -- Replay memory
        batch_size {int} -- Number of observations to use per batch step
        gamma {float} -- Reward discount factor
    """
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def load_model(path: str, device: torch.device):
    model = Policy(n_inputs=14, n_outputs=6).to(device)
    model_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def select_model_action(
    device: torch.device, model: Policy, state: torch.tensor, eps: float
) -> Tuple[torch.tensor, bool]:

    sample = random.random()
    if sample > eps:
        return model.act(state), False
    else:
        return (
            torch.tensor(
                [[random.randrange(0, 6)]],
                device=device,
                dtype=torch.long,
            ),
            True,
        )
