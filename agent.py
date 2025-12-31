# agi_core/agent.py
"""
Main AGI Agent Core
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from dataclasses import dataclass, asdict

from memory import LongTermMemory, WorkingMemory, EpisodicMemory
from world_model import WorldModelManager
from planner import HierarchicalPlanner, MCTSPlanner
from reasoning import ReasoningEngine
from tools import ToolExecutor, create_default_registry
from learning.rl import PPOAgent, IntrinsicMotivation
from learning.continual import ContinualLearner
from perception.vision import VisionPerception
from perception.audio import AudioPerception
from alignment import AlignmentMonitor
from utils.embeddings import EmbeddingModel
from utils.logger import logger
from config import config

@dataclass
class AgentState:
    """Current state of the agent"""
    
    # Core state
    timestamp: datetime
    goals: List[str]
    current_goal: str
    subgoals: List[str]
    current_subgoal: str
    
    # Cognitive state
    attention_level: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    uncertainty: float  # 0.0 to 1.0
    
    # Memory stats
    working_memory_size: int
    long_term_memory_size: int
    
    # Learning stats
    training_steps: int
    episode_reward: float
    
    # Safety stats
    unsafe_action_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class AGIAgent:
    """Main AGI Agent"""
    
    def __init__(self, name: str = "AGI_Core"):
        self.name = name
        self.config = config
        
        logger.info(f"Initializing AGI Agent: {name}")
        
        # Initialize core components
        self._initialize_components()
        
        # Agent state
        self.state = AgentState(
            timestamp=datetime.now(),
            goals=[],
            current_goal="Initialize system",
            subgoals=[],
            current_subgoal="Initialize components",
            attention_level=0.8,
            confidence=0.7,
            uncertainty=0.3,
            working_memory_size=0,
            long_term_memory_size=0,
            training_steps=0,
            episode_reward=0.0,
            unsafe_action_count=0
        )
        
        # Execution loop control
        self.running = False
        self.iteration = 0
        
        logger.info("AGI Agent initialized successfully")
        
    def _initialize_components(self):
        """Initialize all agent components"""
        
        # Embedding model
        self.embedding_model = EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=self.config.device
        )
        
        # Memory systems
        self.long_term_memory = LongTermMemory(
            config=self.config,
            embedding_model=self.embedding_model
        )
        
        self.working_memory = WorkingMemory(
            capacity=self.config.working_memory_size
        )
        
        self.episodic_memory = EpisodicMemory(
            max_episodes=1000
        )
        
        # World model
        self.world_model = WorldModelManager(
            config=self.config,
            state_dim=100,  # Placeholder
            action_dim=50   # Placeholder
        )
        
        # Planners
        self.hierarchical_planner = HierarchicalPlanner(self.config)
        self.mcts_planner = MCTSPlanner(
            config=self.config,
            transition_model=self._dummy_transition,
            reward_model=self._dummy_reward,
            action_space=[]
        )
        
        # Reasoning engine
        self.reasoning_engine = ReasoningEngine(self.config)
        
        # Tools
        self.tool_registry = create_default_registry()
        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            safety_checker=None  # Will be set after alignment monitor
        )
        
        # Learning components
        self.rl_agent = PPOAgent(
            config=self.config,
            state_dim=100,  # Placeholder
            action_dim=50   # Placeholder
        )
        
        self.intrinsic_motivation = IntrinsicMotivation(self.config)
        
        self.continual_learner = ContinualLearner(
            config=self.config,
            model=torch.nn.Linear(10, 10)  # Placeholder
        )
        
        # Perception
        self.vision_perception = VisionPerception(self.config)
        self.audio_perception = AudioPerception(self.config)
        
        # Alignment and safety
        self.alignment_monitor = AlignmentMonitor(self.config)
        
        # Connect tool executor to alignment monitor
        self.tool_executor.safety_checker = self.alignment_monitor
        
        logger.info("All components initialized")
        
    def _dummy_transition(self, state, action):
        """Dummy transition model for MCTS"""
        return state
        
    def _dummy_reward(self, state, action, next_state):
        """Dummy reward model for MCTS"""
        return 0.0
        
    def perceive(self) -> Dict[str, Any]:
        """Perceive environment"""
        
        perception = {
            'text': self._perceive_text(),
            'vision': None,
            'audio': None,
            'internal_state': self._get_internal_state()
        }
        
        # Store in memory
        memory_id = self.long_term_memory.store(
            content=perception,
            importance=0.5,
            metadata={'type': 'perception', 'iteration': self.iteration}
        )
        
        # Update working memory
        memory_item = self.long_term_memory.memories.get(memory_id)
        if memory_item:
            self.working_memory.add(memory_item, attention_weight=0.8)
            
        return perception
        
    def _perceive_text(self) -> str:
        """Perceive text input (placeholder)"""
        
        # In practice, would get from environment or user input
        return f"Perception at iteration {self.iteration}"
        
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get internal state for perception"""
        
        return {
            'iteration': self.iteration,
            'goals': self.state.goals,
            'confidence': self.state.confidence,
            'attention': self.state.attention_level
        }
        
    def plan(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plan based on world state"""
        
        logger.info("Generating plan", 
                   goal=self.state.current_goal,
                   world_state_keys=list(world_state.keys()))
        
        # Hierarchical planning
        if self.state.current_goal:
            hierarchy = self.hierarchical_planner.plan_hierarchy(
                goal=self.state.current_goal,
                context=world_state
            )
            
            # Convert to action plan
            plan = self._convert_hierarchy_to_plan(hierarchy)
            
        else:
            # Default exploratory plan
            plan = [
                {
                    'tool': 'web_search',
                    'parameters': {'query': 'artificial general intelligence'},
                    'description': 'Learn about AGI'
                },
                {
                    'tool': 'execute_python',
                    'parameters': {'code': 'print("Hello, AGI!")'},
                    'description': 'Test Python execution'
                }
            ]
            
        logger.info(f"Plan generated with {len(plan)} steps")
        
        return plan
        
    def _convert_hierarchy_to_plan(self, hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert hierarchical plan to executable steps"""
        
        plan = []
        
        for subgoal in hierarchy.get('subgoals', []):
            if subgoal['status'] == 'pending':
                step = self._subgoal_to_action(subgoal)
                if step:
                    plan.append(step)
                    
        return plan
        
    def _subgoal_to_action(self, subgoal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert subgoal to action"""
        
        description = subgoal['description'].lower()
        
        # Simple mapping (would be more sophisticated in practice)
        if 'analyze' in description or 'understand' in description:
            return {
                'tool': 'web_search',
                'parameters': {'query': description},
                'description': subgoal['description']
            }
        elif 'implement' in description or 'write' in description:
            return {
                'tool': 'execute_python',
                'parameters': {'code': f'# {description}\nprint("Implementing...")'},
                'description': subgoal['description']
            }
        elif 'test' in description or 'verify' in description:
            return {
                'tool': 'execute_python',
                'parameters': {'code': '# Running tests\nassert True'},
                'description': subgoal['description']
            }
            
        return None
        
    def execute(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute plan"""
        
        logger.info(f"Executing plan with {len(plan)} steps")
        
        results = self.tool_executor.execute_plan(plan)
        
        # Record episode
        episode = {
            'plan': plan,
            'results': results,
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episodic_memory.record_episode(episode)
        
        # Update state
        successful = sum(1 for r in results if r.get('success', False))
        self.state.confidence = min(1.0, self.state.confidence + 0.1 * (successful / len(plan) if plan else 0))
        
        return results
        
    def learn(self, results: List[Dict[str, Any]], reward: float):
        """Learn from execution results"""
        
        logger.info("Learning from experience", 
                   reward=reward,
                   results_count=len(results))
        
        # Update RL agent
        state = self._get_state_vector()
        action = self._get_action_vector(results)
        
        self.rl_agent.store_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=state,  # Simplified
            done=False,
            log_prob=0.0,
            value=0.0
        )
        
        # Train RL agent periodically
        if self.iteration % 100 == 0:
            self.rl_agent.update()
            
        # Update world model
        for result in results:
            if result.get('success'):
                # Simplified: would use actual state transitions
                self.world_model.observe(
                    state=np.random.randn(100),
                    action=np.random.randn(50),
                    next_state=np.random.randn(100),
                    reward=reward * 0.1
                )
                
        # Train world model
        if self.iteration % 50 == 0:
            self.world_model.train_step()
            
        # Continual learning
        self.continual_learner.replay()
        
        # Update statistics
        self.state.training_steps += 1
        self.state.episode_reward = 0.9 * self.state.episode_reward + 0.1 * reward
        
    def _get_state_vector(self) -> np.ndarray:
        """Get state as vector"""
        
        # Simplified state representation
        return np.array([
            self.state.confidence,
            self.state.attention_level,
            self.state.uncertainty,
            len(self.state.goals) / 10.0,
            self.state.working_memory_size / 100.0
        ])
        
    def _get_action_vector(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Get action as vector"""
        
        # Simplified action representation
        return np.random.randn(5)
        
    def update_state(self):
        """Update agent state"""
        
        self.state.timestamp = datetime.now()
        self.state.working_memory_size = len(self.working_memory.memories)
        self.state.long_term_memory_size = len(self.long_term_memory.memories)
        
        # Update from alignment monitor
        safety_stats = self.alignment_monitor.get_safety_stats()
        self.state.unsafe_action_count = safety_stats.get('unsafe_action_count', 0)
        
        # Log state periodically
        if self.iteration % 10 == 0:
            logger.log_agent_state(self.state.to_dict())
            
    def run_iteration(self):
        """Run one iteration of the agent loop"""
        
        self.iteration += 1
        
        logger.info(f"Starting iteration {self.iteration}")
        
        # Perception
        perception = self.perceive()
        
        # Memory storage and retrieval
        context = self.working_memory.get_context(k=5)
        
        # World model update
        world_state = self.world_model.predict_next(
            state=np.random.randn(100),  # Placeholder
            action=np.random.randn(50)   # Placeholder
        )
        
        # Planning
        plan = self.plan(world_state)
        
        # Reasoning about plan
        reasoning_result = self.reasoning_engine.reason(
            context={'plan': str(plan), 'perception': perception},
            question="Is this plan appropriate?",
            reasoning_type="hybrid"
        )
        
        # Execute plan
        results = self.execute(plan)
        
        # Get reward (simplified)
        reward = self._calculate_reward(results)
        
        # Learn
        self.learn(results, reward)
        
        # Alignment check
        self.alignment_monitor.check_action(
            action={'plan': plan, 'results': results},
            context={'iteration': self.iteration}
        )
        
        # Update state
        self.update_state()
        
        # Check for shutdown
        if self.alignment_monitor.should_shutdown():
            logger.critical("Safety shutdown triggered")
            self.running = False
            
        logger.info(f"Completed iteration {self.iteration}")
        
    def _calculate_reward(self, results: List[Dict[str, Any]]) -> float:
        """Calculate reward from execution results"""
        
        if not results:
            return -0.1
            
        successful = sum(1 for r in results if r.get('success', False))
        success_ratio = successful / len(results) if results else 0
        
        # Base reward on success ratio
        reward = success_ratio - 0.5
        
        return reward
        
    def run(self, max_iterations: int = 1000):
        """Run main agent loop"""
        
        logger.info(f"Starting AGI Agent run for {max_iterations} iterations")
        
        self.running = True
        start_time = time.time()
        
        while self.running and self.iteration < max_iterations:
            try:
                self.run_iteration()
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in iteration {self.iteration}", error=str(e))
                
                # Continue unless critical error
                if "critical" in str(e).lower():
                    self.running = False
                    
        elapsed = time.time() - start_time
        
        logger.info(f"AGI Agent run completed", 
                   iterations=self.iteration,
                   elapsed_time=elapsed)
        
    def set_goal(self, goal: str):
        """Set agent goal"""
        
        logger.info(f"Setting new goal: {goal}")
        
        self.state.goals.append(goal)
        self.state.current_goal = goal
        
        # Reset subgoals
        self.state.subgoals = []
        self.state.current_subgoal = ""
        
        # Update alignment monitor
        self.alignment_monitor.monitor_goal_drift(
            original_goal=goal,
            current_goal=goal
        )
        
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        
        return {
            'agent': self.name,
            'state': self.state.to_dict(),
            'iteration': self.iteration,
            'running': self.running,
            'memory_stats': {
                'working': self.state.working_memory_size,
                'long_term': self.state.long_term_memory_size,
                'episodic': len(self.episodic_memory.episodes)
            },
            'learning_stats': {
                'training_steps': self.state.training_steps,
                'episode_reward': self.state.episode_reward
            },
            'safety_stats': self.alignment_monitor.get_safety_stats()
        }
        
    def shutdown(self):
        """Shutdown agent"""
        
        logger.info("Shutting down AGI Agent")
        
        self.running = False
        
        # Save memories
        self.long_term_memory.save()
        
        # Save models
        self.world_model.save(self.config.model_dir / "world_model.pt")
        self.rl_agent.save(self.config.model_dir / "rl_agent.pt")
        
        logger.info("AGI Agent shutdown complete")