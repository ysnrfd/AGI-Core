# agi_core/config.py
"""
Configuration management for AGI Core
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import torch

@dataclass
class AGIConfig:
    """Main configuration class"""
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_workers: int = 4
    log_level: str = "INFO"
    
    # Memory
    memory_dim: int = 768
    memory_max_size: int = 100000
    working_memory_size: int = 100
    similarity_threshold: float = 0.8
    
    # Planning
    planning_horizon: int = 5
    beam_width: int = 3
    max_plan_depth: int = 10
    
    # Learning
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 0.005
    
    # World Model
    world_model_hidden: int = 512
    prediction_horizon: int = 10
    
    # Safety
    safety_threshold: float = 0.7
    max_unsafe_actions: int = 3
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    model_dir: Path = field(default_factory=lambda: Path("./models"))
    memory_path: Path = field(default_factory=lambda: Path("./memory"))
    
    def __post_init__(self):
        """Create directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.memory_path.mkdir(exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'AGIConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v if not isinstance(v, Path) else str(v) 
                for k, v in self.__dict__.items()}

config = AGIConfig()