# agi_core/planner.py
"""
Hierarchical planning system with MCTS and LLM reasoning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import random
from dataclasses import dataclass

from utils.logger import logger

@dataclass
class PlanNode:
    """Node in planning tree"""
    state: Any
    action: Optional[Any] = None
    parent: Optional['PlanNode'] = None
    children: List['PlanNode'] = None
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0
    reward: float = 0.0
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
            
    def ucb_score(self, exploration_weight: float = 1.41) -> float:
        """Calculate UCB score"""
        if self.visits == 0:
            return float('inf')
            
        exploitation = self.value / self.visits
        exploration = exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration + self.prior
        
    def is_leaf(self) -> bool:
        """Check if node is leaf"""
        return len(self.children) == 0
        
    def add_child(self, child: 'PlanNode'):
        """Add child node"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

class MCTSPlanner:
    """Monte Carlo Tree Search planner"""
    
    def __init__(self, config, transition_model: Callable, 
                 reward_model: Callable, action_space: List[Any]):
        self.config = config
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.action_space = action_space
        
        # MCTS parameters
        self.max_iterations = 1000
        self.max_depth = config.max_plan_depth
        self.exploration_weight = 1.41
        self.simulations_per_action = 10
        
    def plan(self, initial_state: Any, goal_state: Any = None, 
             horizon: int = 5) -> List[Any]:
        """Generate plan using MCTS"""
        
        logger.info("Starting MCTS planning", 
                   horizon=horizon,
                   action_space_size=len(self.action_space))
        
        # Root node
        root = PlanNode(state=initial_state)
        
        # MCTS iterations
        for iteration in range(self.max_iterations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_leaf() and node.depth < self.max_depth:
                node = self._expand(node)
                
            # Simulation
            value = self._simulate(node, goal_state, horizon)
            
            # Backpropagation
            self._backpropagate(node, value)
            
        # Get best plan
        best_plan = self._get_best_plan(root)
        
        logger.info("MCTS planning completed", 
                   iterations=self.max_iterations,
                   plan_length=len(best_plan))
        
        return best_plan
        
    def _select(self, node: PlanNode) -> PlanNode:
        """Select node using UCB"""
        
        while not node.is_leaf():
            if len(node.children) < len(self.action_space):
                # Not fully expanded
                return node
                
            # Select child with highest UCB
            scores = [child.ucb_score(self.exploration_weight) for child in node.children]
            node = node.children[np.argmax(scores)]
            
        return node
        
    def _expand(self, node: PlanNode) -> PlanNode:
        """Expand node with new action"""
        
        # Get untried actions
        tried_actions = {child.action for child in node.children}
        untried_actions = [a for a in self.action_space if a not in tried_actions]
        
        if not untried_actions:
            return node
            
        # Select random untried action
        action = random.choice(untried_actions)
        
        # Simulate transition
        next_state = self.transition_model(node.state, action)
        reward = self.reward_model(node.state, action, next_state)
        
        # Create child node
        child = PlanNode(
            state=next_state,
            action=action,
            reward=reward,
            prior=1.0 / len(self.action_space)  # Uniform prior
        )
        
        node.add_child(child)
        
        return child
        
    def _simulate(self, node: PlanNode, goal_state: Any, horizon: int) -> float:
        """Simulate rollout from node"""
        
        state = node.state
        total_reward = 0.0
        
        for step in range(horizon):
            # Select random action
            action = random.choice(self.action_space)
            
            # Simulate transition
            next_state = self.transition_model(state, action)
            reward = self.reward_model(state, action, next_state)
            
            total_reward += reward * (self.config.gamma ** step)
            
            # Update state
            state = next_state
            
            # Early termination if goal reached
            if goal_state is not None and self._is_goal_reached(state, goal_state):
                total_reward += 100.0  # Goal bonus
                break
                
        return total_reward
        
    def _backpropagate(self, node: PlanNode, value: float):
        """Backpropagate value through tree"""
        
        while node is not None:
            node.visits += 1
            node.value += value
            
            # Discount value for parent
            value = node.reward + self.config.gamma * value
            
            node = node.parent
            
    def _get_best_plan(self, root: PlanNode) -> List[Any]:
        """Extract best plan from tree"""
        
        plan = []
        node = root
        
        while node.children:
            # Select child with highest value/visits ratio
            best_child = max(node.children, 
                           key=lambda c: c.value / (c.visits + 1e-8))
            
            if best_child.action is not None:
                plan.append(best_child.action)
                
            node = best_child
            
        return plan
        
    def _is_goal_reached(self, state: Any, goal_state: Any) -> bool:
        """Check if goal state is reached"""
        # Simple equality check - can be customized
        return state == goal_state

class HierarchicalPlanner:
    """Hierarchical goal decomposition planner"""
    
    def __init__(self, config):
        self.config = config
        self.subgoal_generator = None
        self.task_hierarchy = {}
        
    def decompose_goal(self, goal: str) -> List[str]:
        """Decompose high-level goal into subgoals"""
        
        # Simple rule-based decomposition
        # In practice, this would use an LLM or learned model
        
        subgoals = []
        
        # Example decomposition rules
        if "write" in goal.lower() and "code" in goal.lower():
            subgoals = [
                "Analyze requirements",
                "Design architecture",
                "Implement core functions",
                "Write tests",
                "Debug and optimize"
            ]
        elif "analyze" in goal.lower() and "data" in goal.lower():
            subgoals = [
                "Load and preprocess data",
                "Perform exploratory analysis",
                "Apply statistical tests",
                "Create visualizations",
                "Draw conclusions"
            ]
        else:
            # Default decomposition
            subgoals = [
                "Understand the problem",
                "Gather information",
                "Generate solution options",
                "Evaluate alternatives",
                "Execute chosen solution",
                "Verify results"
            ]
            
        return subgoals
        
    def plan_hierarchy(self, goal: str, context: Dict) -> Dict[str, Any]:
        """Create hierarchical plan"""
        
        logger.info("Creating hierarchical plan", goal=goal)
        
        # Generate subgoals
        subgoals = self.decompose_goal(goal)
        
        # Create plan hierarchy
        plan = {
            'goal': goal,
            'subgoals': [],
            'actions': [],
            'dependencies': {},
            'estimated_duration': len(subgoals) * 60  # minutes
        }
        
        for i, subgoal in enumerate(subgoals):
            subplan = {
                'id': f"subgoal_{i}",
                'description': subgoal,
                'priority': len(subgoals) - i,  # Earlier subgoals higher priority
                'estimated_time': 60,  # minutes
                'prerequisites': [] if i == 0 else [f"subgoal_{i-1}"],
                'status': 'pending'
            }
            
            plan['subgoals'].append(subplan)
            
        logger.info("Hierarchical plan created", 
                   subgoal_count=len(subgoals))
        
        return plan
        
    def update_plan(self, plan: Dict, progress: Dict) -> Dict:
        """Update plan based on progress"""
        
        # Update subgoal status
        for subgoal in plan['subgoals']:
            if subgoal['id'] in progress.get('completed_subgoals', []):
                subgoal['status'] = 'completed'
            elif subgoal['id'] in progress.get('failed_subgoals', []):
                subgoal['status'] = 'failed'
            elif subgoal['id'] in progress.get('active_subgoals', []):
                subgoal['status'] = 'active'
                
        # Recalculate estimated duration
        remaining = [s for s in plan['subgoals'] if s['status'] != 'completed']
        plan['estimated_duration'] = len(remaining) * 60
        
        return plan

class BeamSearchPlanner:
    """Beam search planner for efficient planning"""
    
    def __init__(self, config, beam_width: int = 3):
        self.config = config
        self.beam_width = beam_width
        
    def plan(self, initial_state: Any, goal_checker: Callable,
             transition: Callable, heuristic: Callable,
             max_depth: int = 10) -> List[Any]:
        """Plan using beam search"""
        
        # Beam: list of (path, state, score)
        beam = [([], initial_state, 0.0)]
        
        for depth in range(max_depth):
            next_beam = []
            
            for path, state, score in beam:
                # Check if goal reached
                if goal_checker(state):
                    return path
                    
                # Generate successors
                # This would use the transition model in practice
                successors = self._generate_successors(state, transition)
                
                for action, next_state in successors:
                    new_path = path + [action]
                    new_score = score + heuristic(next_state)
                    
                    next_beam.append((new_path, next_state, new_score))
                    
            # Keep top-k
            next_beam.sort(key=lambda x: x[2], reverse=True)
            beam = next_beam[:self.beam_width]
            
            if not beam:
                break
                
        # Return best path found
        if beam:
            return beam[0][0]
        else:
            return []
            
    def _generate_successors(self, state: Any, transition: Callable) -> List[Tuple[Any, Any]]:
        """Generate successor states"""
        # Placeholder - would use action space and transition model
        return []