# agi_core/learning/continual.py
"""
Continual learning with Elastic Weight Consolidation and replay buffers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import deque
import copy

from utils.logger import logger

class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""
    
    def __init__(self, model: nn.Module, importance: float = 1000):
        self.model = model
        self.importance = importance
        
        # Store initial parameters
        self.initial_params = {}
        self.fisher_matrix = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone()
                self.fisher_matrix[name] = torch.zeros_like(param.data)
                
    def compute_fisher(self, dataloader, num_samples: int = 100):
        """Compute Fisher information matrix"""
        
        logger.info("Computing Fisher information matrix", 
                   num_samples=num_samples)
        
        # Reset Fisher matrix
        for name in self.fisher_matrix:
            self.fisher_matrix[name].zero_()
            
        # Compute gradients for each sample
        self.model.train()
        
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Forward pass
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_matrix[name] += param.grad.data ** 2 / num_samples
                    
        self.model.eval()
        
        logger.info("Fisher information computed")
        
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC loss"""
        
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_matrix:
                # Quadratic penalty based on Fisher information
                fisher = self.fisher_matrix[name]
                diff = param - self.initial_params[name]
                ewc_loss += (fisher * diff * diff).sum()
                
        return self.importance * ewc_loss
        
    def update_initial_params(self):
        """Update initial parameters for next task"""
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone()

class ReplayBuffer:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, Any], priority: float = 1.0):
        """Add experience to buffer"""
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size: int, alpha: float = 0.6, 
               beta: float = 0.4) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample batch with prioritized experience replay"""
        
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Get experiences
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                
    def clear(self):
        """Clear buffer"""
        
        self.buffer.clear()
        self.priorities.clear()

class ContinualLearner:
    """Main continual learning module"""
    
    def __init__(self, config, model: nn.Module):
        self.config = config
        self.model = model
        
        # EWC
        self.ewc = ElasticWeightConsolidation(model)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=5000)
        
        # Task memory
        self.tasks = []
        self.current_task = 0
        
        # Consolidation scheduler
        self.consolidation_interval = 1000
        
    def learn_task(self, task_data, task_id: int):
        """Learn a new task"""
        
        logger.info(f"Learning task {task_id}", 
                   data_size=len(task_data))
        
        # Store task
        self.tasks.append({
            'id': task_id,
            'data': task_data,
            'fisher': None
        })
        
        self.current_task = task_id
        
        # Train on task data
        self._train_on_task(task_data)
        
        # Consolidate knowledge
        if task_id > 0:
            self.consolidate()
            
    def _train_on_task(self, task_data, epochs: int = 10):
        """Train on task data"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.config.learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in task_data:
                inputs, targets = batch
                
                # Forward pass
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, targets)
                
                # EWC loss
                ewc_loss = self.ewc.compute_ewc_loss()
                
                # Total loss
                loss = task_loss + ewc_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.debug(f"Task training epoch {epoch}", 
                        loss=total_loss / len(task_data))
            
    def consolidate(self):
        """Consolidate knowledge from all tasks"""
        
        logger.info("Consolidating knowledge from all tasks")
        
        # For each previous task
        for task in self.tasks[:-1]:  # Exclude current task
            if task['fisher'] is None:
                # Compute Fisher information for task
                self.ewc.compute_fisher(task['data'])
                task['fisher'] = copy.deepcopy(self.ewc.fisher_matrix)
                
        # Update EWC with consolidated Fisher information
        consolidated_fisher = {}
        
        for task in self.tasks[:-1]:
            fisher = task['fisher']
            for name in fisher:
                if name not in consolidated_fisher:
                    consolidated_fisher[name] = fisher[name]
                else:
                    consolidated_fisher[name] += fisher[name]
                    
        # Average Fisher information
        for name in consolidated_fisher:
            consolidated_fisher[name] /= max(len(self.tasks) - 1, 1)
            
        self.ewc.fisher_matrix = consolidated_fisher
        
        # Update initial parameters
        self.ewc.update_initial_params()
        
    def replay(self, batch_size: int = 32):
        """Replay from buffer to prevent forgetting"""
        
        if len(self.replay_buffer.buffer) < batch_size:
            return
            
        # Sample from replay buffer
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        
        if not samples:
            return
            
        # Prepare batch
        inputs = []
        targets = []
        
        for sample in samples:
            inputs.append(sample['input'])
            targets.append(sample['target'])
            
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        # Training step
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.config.learning_rate * 0.1)
        
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Weight loss by importance sampling weights
        loss = (loss * torch.tensor(weights).float()).mean()
        
        # EWC loss
        ewc_loss = self.ewc.compute_ewc_loss()
        
        total_loss = loss + ewc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update priorities based on loss
        with torch.no_grad():
            losses = F.cross_entropy(outputs, targets, reduction='none')
            new_priorities = losses.cpu().numpy() + 1e-6
            
        self.replay_buffer.update_priorities(indices, new_priorities)
        
    def remember(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """Remember experience for replay"""
        
        experience = {
            'input': input_tensor,
            'target': target_tensor,
            'task': self.current_task
        }
        
        # Compute priority based on prediction error
        with torch.no_grad():
            output = self.model(input_tensor.unsqueeze(0))
            loss = F.cross_entropy(output, target_tensor.unsqueeze(0))
            priority = loss.item() + 1e-6
            
        self.replay_buffer.add(experience, priority)
        
    def evaluate_forgetting(self, test_data: Dict[int, Any]) -> Dict[int, float]:
        """Evaluate forgetting on previous tasks"""
        
        results = {}
        
        self.model.eval()
        
        with torch.no_grad():
            for task_id, data in test_data.items():
                if task_id >= self.current_task:
                    continue
                    
                task_loss = 0
                correct = 0
                total = 0
                
                for inputs, targets in data:
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    
                    task_loss += loss.item()
                    
                    # Accuracy
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)
                    
                accuracy = 100. * correct / total if total > 0 else 0
                
                results[task_id] = {
                    'loss': task_loss / len(data),
                    'accuracy': accuracy
                }
                
        self.model.train()
        
        return results