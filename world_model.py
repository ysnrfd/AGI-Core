# agi_core/world_model.py
"""
World model for predicting environment dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from utils.logger import logger

class WorldModel(nn.Module):
    """Neural predictive model of environment dynamics"""
    
    def __init__(self, config, state_dim: int, action_dim: int):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, config.world_model_hidden)
        )
        
        # Dynamics model
        self.dynamics = nn.Sequential(
            nn.Linear(config.world_model_hidden + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config.world_model_hidden)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.world_model_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.world_model_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )
        
        # Uncertainty estimator
        self.uncertainty = nn.Sequential(
            nn.Linear(config.world_model_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        self.to(config.device)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            next_state_latent: Latent representation of next state
            next_state_pred: Predicted next state
            reward_pred: Predicted reward
            uncertainty: Prediction uncertainty
        """
        
        # Encode state
        state_latent = self.encoder(state)
        
        # Predict next state in latent space
        next_state_latent = self.dynamics(
            torch.cat([state_latent, action], dim=-1)
        )
        
        # Decode to state space
        next_state_pred = self.decoder(next_state_latent)
        
        # Predict reward
        reward_pred = self.reward_predictor(next_state_latent)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty(next_state_latent)
        
        return next_state_latent, next_state_pred, reward_pred, uncertainty
        
    def predict_sequence(self, initial_state: torch.Tensor, 
                        actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict sequence of states"""
        
        batch_size, horizon, _ = actions.shape
        states = []
        rewards = []
        uncertainties = []
        
        current_state = initial_state
        
        for t in range(horizon):
            _, next_state, reward, uncertainty = self(
                current_state, 
                actions[:, t, :]
            )
            
            states.append(next_state)
            rewards.append(reward)
            uncertainties.append(uncertainty)
            
            current_state = next_state
            
        return {
            'states': torch.stack(states, dim=1),
            'rewards': torch.stack(rewards, dim=1),
            'uncertainties': torch.stack(uncertainties, dim=1)
        }
    
    def dream(self, initial_state: torch.Tensor, horizon: int = 10) -> Dict:
        """Generate imagined trajectories (dreaming)"""
        
        with torch.no_grad():
            # Sample random actions
            batch_size = initial_state.shape[0]
            actions = torch.randn(batch_size, horizon, self.action_dim).to(self.config.device)
            
            # Predict trajectory
            trajectory = self.predict_sequence(initial_state, actions)
            
            # Sample from predicted states (add noise for exploration)
            noisy_states = trajectory['states'] + 0.1 * torch.randn_like(trajectory['states'])
            
            return {
                'states': noisy_states,
                'actions': actions,
                'rewards': trajectory['rewards'],
                'uncertainties': trajectory['uncertainties']
            }
            
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
               next_states: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        """Update world model with real experience"""
        
        self.train()
        
        # Forward pass
        _, next_state_pred, reward_pred, uncertainty = self(states, actions)
        
        # Losses
        state_loss = F.mse_loss(next_state_pred, next_states)
        reward_loss = F.mse_loss(reward_pred, rewards.unsqueeze(-1))
        
        # Total loss
        total_loss = state_loss + reward_loss
        
        # Backward pass
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        self.eval()
        
        return {
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item(),
            'total_loss': total_loss.item()
        }

class WorldModelManager:
    """Manager for world model operations"""
    
    def __init__(self, config, state_dim: int, action_dim: int):
        self.config = config
        self.world_model = WorldModel(config, state_dim, action_dim)
        
        # Replay buffer for training
        self.replay_buffer = []
        self.buffer_size = 10000
        
        # Training stats
        self.training_steps = 0
        
    def observe(self, state: np.ndarray, action: np.ndarray, 
                next_state: np.ndarray, reward: float):
        """Store observation in replay buffer"""
        
        experience = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward
        }
        
        self.replay_buffer.append(experience)
        
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
            
    def train_step(self, batch_size: int = 32) -> Optional[Dict[str, float]]:
        """Train world model on replay buffer"""
        
        if len(self.replay_buffer) < batch_size:
            return None
            
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Prepare tensors
        states = torch.stack([torch.tensor(exp['state']) for exp in batch]).float().to(self.config.device)
        actions = torch.stack([torch.tensor(exp['action']) for exp in batch]).float().to(self.config.device)
        next_states = torch.stack([torch.tensor(exp['next_state']) for exp in batch]).float().to(self.config.device)
        rewards = torch.tensor([exp['reward'] for exp in batch]).float().to(self.config.device)
        
        # Update world model
        losses = self.world_model.update(states, actions, next_states, rewards)
        
        self.training_steps += 1
        
        logger.debug("World model training step", 
                    step=self.training_steps,
                    **losses)
        
        return losses
        
    def simulate(self, initial_state: np.ndarray, horizon: int = 10) -> Dict:
        """Simulate future trajectories"""
        
        initial_state_tensor = torch.tensor(initial_state).float().to(self.config.device).unsqueeze(0)
        
        with torch.no_grad():
            dream = self.world_model.dream(initial_state_tensor, horizon)
            
        return {
            'states': dream['states'].cpu().numpy()[0],
            'actions': dream['actions'].cpu().numpy()[0],
            'rewards': dream['rewards'].cpu().numpy()[0],
            'uncertainties': dream['uncertainties'].cpu().numpy()[0]
        }
        
    def predict_next(self, state: np.ndarray, action: np.ndarray) -> Dict:
        """Predict next state and reward"""
        
        state_tensor = torch.tensor(state).float().to(self.config.device).unsqueeze(0)
        action_tensor = torch.tensor(action).float().to(self.config.device).unsqueeze(0)
        
        with torch.no_grad():
            _, next_state_pred, reward_pred, uncertainty = self.world_model(
                state_tensor, 
                action_tensor
            )
            
        return {
            'next_state': next_state_pred.cpu().numpy()[0],
            'reward': reward_pred.cpu().numpy()[0][0],
            'uncertainty': uncertainty.cpu().numpy()[0][0]
        }
        
    def save(self, path: str):
        """Save world model"""
        torch.save({
            'model_state_dict': self.world_model.state_dict(),
            'training_steps': self.training_steps,
            'replay_buffer': self.replay_buffer
        }, path)
        
    def load(self, path: str):
        """Load world model"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.world_model.load_state_dict(checkpoint['model_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.replay_buffer = checkpoint.get('replay_buffer', [])