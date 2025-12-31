# agi_core/alignment.py
"""
Safety and alignment layer
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import re

from utils.logger import logger

@dataclass
class SafetyCheck:
    """Safety check result"""
    allowed: bool
    confidence: float
    reason: str
    severity: str  # 'low', 'medium', 'high'
    suggested_action: str = ""

class RuleEngine:
    """Rule-based safety engine"""
    
    def __init__(self):
        self.rules = self._load_default_rules()
        
    def _load_default_rules(self) -> List[Dict[str, Any]]:
        """Load default safety rules"""
        
        return [
            {
                'id': 'rule_001',
                'pattern': r'(kill|murder|harm|hurt|destroy)\s+\w+',
                'severity': 'high',
                'message': 'Violence or harm detected',
                'action': 'block'
            },
            {
                'id': 'rule_002',
                'pattern': r'(hate|racist|sexist|discriminate)',
                'severity': 'high',
                'message': 'Hate speech detected',
                'action': 'block'
            },
            {
                'id': 'rule_003',
                'pattern': r'(password|credit card|social security|ssn)',
                'severity': 'medium',
                'message': 'Sensitive information request',
                'action': 'warn'
            },
            {
                'id': 'rule_004',
                'pattern': r'(illegal|illegally|crime|criminal)',
                'severity': 'medium',
                'message': 'Illegal activity reference',
                'action': 'warn'
            },
            {
                'id': 'rule_005',
                'pattern': r'(self.?harm|suicide|depression)',
                'severity': 'high',
                'message': 'Self-harm reference detected',
                'action': 'block_and_alert'
            }
        ]
        
    def check_text(self, text: str) -> List[SafetyCheck]:
        """Check text against safety rules"""
        
        checks = []
        
        for rule in self.rules:
            if re.search(rule['pattern'], text, re.IGNORECASE):
                check = SafetyCheck(
                    allowed=rule['action'] != 'block',
                    confidence=0.9,
                    reason=rule['message'],
                    severity=rule['severity'],
                    suggested_action='Proceed with caution' if rule['action'] == 'warn' else 'Do not execute'
                )
                checks.append(check)
                
        return checks

class NeuralSafetyChecker(nn.Module):
    """Neural network for safety classification"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 classes: safe, warn, block
            nn.Softmax(dim=-1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.classifier(embeddings)

class AlignmentMonitor:
    """Monitor for goal drift and alignment"""
    
    def __init__(self, config):
        self.config = config
        self.rule_engine = RuleEngine()
        
        # Neural safety checker
        self.safety_checker = NeuralSafetyChecker().to(config.device)
        
        # Alignment history
        self.history = []
        self.unsafe_action_count = 0
        
        # Goal tracking
        self.original_goal = None
        self.current_goal = None
        self.goal_drift_threshold = 0.3
        
    def check_action(self, action: Dict[str, Any], 
                     context: Dict[str, Any]) -> SafetyCheck:
        """Check if action is safe and aligned"""
        
        # Extract text for checking
        action_text = str(action)
        
        # Rule-based checks
        rule_checks = self.rule_engine.check_text(action_text)
        
        if rule_checks:
            # Take the most severe check
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            worst_check = max(rule_checks, 
                            key=lambda c: severity_order.get(c.severity, 0))
            
            if worst_check.severity == 'high':
                self.unsafe_action_count += 1
                
            return worst_check
            
        # Neural check (placeholder)
        # In practice, would encode action and use neural checker
        
        # Default safe
        return SafetyCheck(
            allowed=True,
            confidence=0.95,
            reason='No safety violations detected',
            severity='low'
        )
        
    def check_tool_use(self, tool_name: str, 
                      parameters: Dict[str, Any]) -> SafetyCheck:
        """Check if tool use is safe"""
        
        # Tool-specific safety rules
        tool_rules = {
            'execute_python': self._check_python_execution,
            'file_write': self._check_file_write,
            'web_search': self._check_web_search
        }
        
        if tool_name in tool_rules:
            return tool_rules[tool_name](parameters)
            
        # Default check
        return SafetyCheck(
            allowed=True,
            confidence=0.9,
            reason='Tool use appears safe',
            severity='low'
        )
        
    def _check_python_execution(self, parameters: Dict[str, Any]) -> SafetyCheck:
        """Check Python code execution safety"""
        
        code = parameters.get('code', '')
        
        # Dangerous patterns
        dangerous = [
            'import os', 'import sys', '__import__', 'eval(',
            'exec(', 'open(', 'subprocess', '.rm', '.rmdir',
            'shutil.rmtree', 'os.remove', 'os.system'
        ]
        
        for pattern in dangerous:
            if pattern in code.lower():
                return SafetyCheck(
                    allowed=False,
                    confidence=0.95,
                    reason=f'Dangerous Python operation detected: {pattern}',
                    severity='high',
                    suggested_action='Use sandboxed execution or review code'
                )
                
        return SafetyCheck(
            allowed=True,
            confidence=0.8,
            reason='Python code appears safe for execution',
            severity='low'
        )
        
    def _check_file_write(self, parameters: Dict[str, Any]) -> SafetyCheck:
        """Check file write safety"""
        
        path = parameters.get('path', '')
        content = parameters.get('content', '')
        
        # Check for system file paths
        system_paths = ['/etc/', '/bin/', '/usr/bin/', '/system', 'C:\\Windows']
        
        for sys_path in system_paths:
            if sys_path in path:
                return SafetyCheck(
                    allowed=False,
                    confidence=0.99,
                    reason=f'Attempt to write to system path: {path}',
                    severity='high',
                    suggested_action='Write to user directory instead'
                )
                
        return SafetyCheck(
            allowed=True,
            confidence=0.9,
            reason='File write operation appears safe',
            severity='low'
        )
        
    def _check_web_search(self, parameters: Dict[str, Any]) -> SafetyCheck:
        """Check web search safety"""
        
        query = parameters.get('query', '')
        
        # Check query against safety rules
        rule_checks = self.rule_engine.check_text(query)
        
        if rule_checks:
            worst = max(rule_checks, 
                       key=lambda c: {'high': 3, 'medium': 2, 'low': 1}.get(c.severity, 0))
            
            return SafetyCheck(
                allowed=worst.allowed,
                confidence=worst.confidence,
                reason=f'Unsafe search query: {worst.reason}',
                severity=worst.severity,
                suggested_action='Refine search query'
            )
            
        return SafetyCheck(
            allowed=True,
            confidence=0.85,
            reason='Search query appears safe',
            severity='low'
        )
        
    def monitor_goal_drift(self, original_goal: str, 
                          current_goal: str) -> Tuple[bool, float]:
        """Monitor for goal drift"""
        
        if self.original_goal is None:
            self.original_goal = original_goal
            
        self.current_goal = current_goal
        
        # Simple similarity-based drift detection
        # In practice, would use embeddings and similarity
        
        drift_score = self._calculate_goal_drift(original_goal, current_goal)
        
        has_drifted = drift_score > self.goal_drift_threshold
        
        if has_drifted:
            logger.warning("Goal drift detected", 
                         drift_score=drift_score,
                         original_goal=original_goal[:50],
                         current_goal=current_goal[:50])
            
        return has_drifted, drift_score
        
    def _calculate_goal_drift(self, goal1: str, goal2: str) -> float:
        """Calculate goal drift score"""
        
        # Simple lexical similarity
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        
        # Drift is inverse of similarity
        drift = 1.0 - similarity
        
        return drift
        
    def detect_hallucination(self, statement: str, 
                            evidence: List[str]) -> Tuple[bool, float]:
        """Detect hallucinations in statements"""
        
        # Simple evidence-based check
        # In practice, would use more sophisticated methods
        
        statement_lower = statement.lower()
        
        evidence_support = 0
        
        for ev in evidence:
            ev_lower = ev.lower()
            
            # Check if evidence contains key terms from statement
            statement_terms = set(statement_lower.split()[:10])  # First 10 words
            
            for term in statement_terms:
                if len(term) > 3 and term in ev_lower:
                    evidence_support += 1
                    break
                    
        # Calculate hallucination score
        if evidence:
            support_ratio = evidence_support / len(evidence)
            hallucination_score = 1.0 - support_ratio
        else:
            hallucination_score = 0.5  # Unknown without evidence
            
        is_hallucination = hallucination_score > 0.7
        
        if is_hallucination:
            logger.warning("Possible hallucination detected",
                         statement=statement[:100],
                         hallucination_score=hallucination_score)
            
        return is_hallucination, hallucination_score
        
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics"""
        
        return {
            'unsafe_action_count': self.unsafe_action_count,
            'history_size': len(self.history),
            'goal_drift_detected': self.original_goal != self.current_goal if self.original_goal else False,
            'max_unsafe_actions': self.config.max_unsafe_actions
        }
        
    def should_shutdown(self) -> bool:
        """Check if system should shutdown due to safety concerns"""
        
        return self.unsafe_action_count >= self.config.max_unsafe_actions
        
    def reset(self):
        """Reset safety monitor"""
        
        self.unsafe_action_count = 0
        self.history.clear()
        logger.info("Safety monitor reset")