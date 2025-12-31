# agi_core/learning/rl.py
"""
Reinforcement Learning module with PPO and intrinsic motivation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import deque

from utils.logger import logger

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared base
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Intrinsic motivation
        self.intrinsic_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        
        features = self.base(state)
        
        # Actor
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        
        # Critic
        value = self.critic(features)
        
        return mean, std, value
        
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action"""
        
        mean, std, value = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            
        log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
        
        return action, log_prob, value
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action"""
        
        mean, std, value = self.forward(state)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(-1)
        entropy = normal.entropy().sum(-1)
        
        return log_prob, entropy, value
        
    def predict_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, 
                                next_state: torch.Tensor) -> torch.Tensor:
        """Predict intrinsic reward (curiosity)"""
        
        predicted_next = self.intrinsic_predictor(torch.cat([state, action], dim=-1))
        
        # Intrinsic reward based on prediction error
        intrinsic_reward = F.mse_loss(predicted_next, next_state, reduction='none').mean(-1)
        
        return intrinsic_reward

class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, config, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy = ActorCritic(state_dim, action_dim).to(config.device)
        self.old_policy = ActorCritic(state_dim, action_dim).to(config.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = 10000
        
        # Training parameters
        self.gamma = config.gamma
        self.tau = config.tau
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.intrinsic_coef = 0.1
        
        # Statistics
        self.episode_rewards = []
        self.training_steps = 0
        
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, 
                        done: bool, log_prob: float, value: float):
        """Store transition in buffer"""
        
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool]) -> np.ndarray:
        """Compute advantages using GAE"""
        
        advantages = []
        returns = []
        gae = 0
        
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * gae * (1 - dones[t])
            advantages.insert(0, gae)
            next_value = values[t]
            
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
        
    def update(self, batch_size: int = 64, epochs: int = 4):
        """Update policy"""
        
        if len(self.buffer) < batch_size:
            return
            
        logger.info("Starting PPO update", 
                   buffer_size=len(self.buffer),
                   batch_size=batch_size)
        
        # Convert buffer to tensors
        states = torch.tensor([t['state'] for t in self.buffer]).float().to(self.config.device)
        actions = torch.tensor([t['action'] for t in self.buffer]).float().to(self.config.device)
        rewards = torch.tensor([t['reward'] for t in self.buffer]).float().to(self.config.device)
        next_states = torch.tensor([t['next_state'] for t in self.buffer]).float().to(self.config.device)
        dones = torch.tensor([t['done'] for t in self.buffer]).float().to(self.config.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.buffer]).float().to(self.config.device)
        values = torch.tensor([t['value'] for t in self.buffer]).float().to(self.config.device)
        
        # Compute advantages
        advantages = self.compute_advantages(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy()
        )
        advantages = torch.tensor(advantages).float().to(self.config.device)
        
        # Compute returns
        returns = advantages + values
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Training epochs
        for epoch in range(epochs):
            # Shuffle indices
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_next_states = next_states[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.policy.evaluate(
                    batch_states, 
                    batch_actions
                )
                
                # Compute intrinsic reward
                intrinsic_reward = self.policy.predict_intrinsic_reward(
                    batch_states,
                    batch_actions,
                    batch_next_states
                )
                
                # Total reward
                total_reward = batch_returns + self.intrinsic_coef * intrinsic_reward
                
                # Policy ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Policy loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, total_reward.unsqueeze(-1))
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                self.training_steps += 1
                
                # Logging
                if self.training_steps % 100 == 0:
                    logger.debug("PPO training step",
                                step=self.training_steps,
                                policy_loss=policy_loss.item(),
                                value_loss=value_loss.item(),
                                entropy=entropy.mean().item())
        
        # Clear buffer
        self.buffer.clear()
        
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action"""
        
        state_tensor = torch.tensor(state).float().to(self.config.device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor, deterministic)
            
        return action.cpu().numpy()[0], log_prob.item(), value.item()
        
    def save(self, path: str):
        """Save agent"""
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards
        }, path)
        
    def load(self, path: str):
        """Load agent"""
        
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        
    def record_episode(self, total_reward: float):
        """Record episode reward"""
        
        self.episode_rewards.append(total_reward)
        
        # Keep only recent episodes
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        
        if not self.episode_rewards:
            return {}
            
        return {
            'training_steps': self.training_steps,
            'mean_reward': np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            'episode_count': len(self.episode_rewards)
        }

class IntrinsicMotivation:
    """Intrinsic motivation mechanisms"""
    
    def __init__(self, config):
        self.config = config
        
        # Curiosity
        self.forward_model = nn.Sequential(
            nn.Linear(768, 512),  # State + action dimension
            nn.ReLU(),
            nn.Linear(512, 768)   # Next state prediction
        ).to(config.device)
        
        # Novelty memory
        self.novelty_memory = deque(maxlen=1000)
        self.novelty_threshold = 0.1
        
    def compute_curiosity(self, state: torch.Tensor, action: torch.Tensor, 
                         next_state: torch.Tensor) -> float:
        """Compute curiosity-based intrinsic reward"""
        
        # Predict next state
        predicted = self.forward_model(torch.cat([state, action], dim=-1))
        
        # Prediction error as curiosity
        prediction_error = F.mse_loss(predicted, next_state)
        
        # Normalize
        curiosity = prediction_error.item()
        
        return curiosity
        
    def compute_novelty(self, state: torch.Tensor) -> float:
        """Compute novelty of state"""
        
        if not self.novelty_memory:
            return 1.0  # Maximum novelty for first state
            
        # Compute similarity to remembered states
        similarities = []
        for mem_state in self.novelty_memory:
            sim = F.cosine_similarity(state, mem_state, dim=-1).item()
            similarities.append(sim)
            
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        # Store state if novel enough
        if novelty > self.novelty_threshold:
            self.novelty_memory.append(state.detach())
            
        return novelty
        
    def compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor,
                               next_state: torch.Tensor) -> float:
        """Compute total intrinsic reward"""
        
        curiosity = self.compute_curiosity(state, action, next_state)
        novelty = self.compute_novelty(state)
        
        # Combine
        intrinsic_reward = 0.7 * curiosity + 0.3 * novelty
        
        return intrinsic_reward