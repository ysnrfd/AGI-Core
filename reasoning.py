# agi_core/reasoning.py
"""
Reasoning engine with symbolic and neural components
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from utils.logger import logger

@dataclass
class ReasoningStep:
    """Step in reasoning chain"""
    step_type: str  # 'observation', 'inference', 'deduction', 'abduction'
    content: str
    confidence: float
    evidence: List[str] = None
    justification: str = ""
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

class NeuralReasoner(nn.Module):
    """Neural reasoning module"""
    
    def __init__(self, config, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.config = config
        
        # Attention-based reasoning
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(3)
        ])
        
        # Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.to(config.device)
        
    def forward(self, inputs: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for neural reasoning"""
        
        # Project input
        x = self.input_proj(inputs)
        
        # Apply attention and reasoning layers
        for layer in self.reasoning_layers:
            x, _ = self.attention(x, x, x, 
                                 key_padding_mask=attention_mask)
            x = layer(x)
            
        # Estimate confidence
        pooled = x.mean(dim=1)
        confidence = self.confidence_net(pooled)
        
        # Project back
        output = self.output_proj(x)
        
        return output, confidence
        
    def reason_about(self, context: List[str], 
                     question: str) -> Tuple[str, float]:
        """Reason about question given context"""
        
        # Encode inputs (placeholder - would use actual encoding)
        context_emb = torch.randn(len(context), 768).to(self.config.device)
        question_emb = torch.randn(1, 768).to(self.config.device)
        
        # Combine
        inputs = torch.cat([context_emb, question_emb], dim=0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output, confidence = self(inputs)
            
        # Generate answer (simplified)
        answer = f"Based on context of {len(context)} items: {question[:50]}..."
        
        return answer, confidence.item()

class SymbolicReasoner:
    """Symbolic reasoning engine"""
    
    def __init__(self):
        self.knowledge_base = []
        self.rules = []
        self.symbols = {}
        
    def add_fact(self, fact: str):
        """Add fact to knowledge base"""
        self.knowledge_base.append(fact)
        
    def add_rule(self, rule: str):
        """Add rule to rule base"""
        self.rules.append(rule)
        
    def deduce(self, query: str) -> Tuple[bool, List[str]]:
        """Deduce if query is true"""
        
        # Simple forward chaining
        proven = []
        agenda = [query]
        
        while agenda:
            current = agenda.pop(0)
            
            # Check if already proven
            if current in proven:
                continue
                
            # Check if in knowledge base
            if current in self.knowledge_base:
                proven.append(current)
                continue
                
            # Try to apply rules
            for rule in self.rules:
                if self._apply_rule(rule, current, proven):
                    proven.append(current)
                    break
                    
        success = query in proven
        return success, proven
        
    def _apply_rule(self, rule: str, goal: str, proven: List[str]) -> bool:
        """Apply rule to prove goal"""
        # Simplified rule application
        # In practice, would use proper logical inference
        
        if "->" in rule:
            antecedent, consequent = rule.split("->")
            
            if consequent.strip() == goal:
                # Check if antecedent is proven
                antecedents = antecedent.split("&")
                if all(a.strip() in proven for a in antecedents):
                    return True
                    
        return False
        
    def explain(self, conclusion: str) -> List[str]:
        """Generate explanation for conclusion"""
        
        _, proof = self.deduce(conclusion)
        
        explanation = [
            f"To prove: {conclusion}",
            "Proof steps:"
        ]
        
        for step in proof:
            explanation.append(f"  - {step}")
            
        explanation.append(f"Conclusion: {conclusion} is {'proven' if conclusion in proof else 'not proven'}")
        
        return explanation

class CausalReasoner:
    """Causal reasoning module"""
    
    def __init__(self):
        self.causal_graph = {}
        self.interventions = {}
        
    def add_causal_relation(self, cause: str, effect: str, strength: float = 1.0):
        """Add causal relation"""
        if cause not in self.causal_graph:
            self.causal_graph[cause] = []
            
        self.causal_graph[cause].append((effect, strength))
        
    def infer_causes(self, effect: str) -> List[Tuple[str, float]]:
        """Infer possible causes for effect"""
        
        causes = []
        for cause, effects in self.causal_graph.items():
            for e, strength in effects:
                if e == effect:
                    causes.append((cause, strength))
                    
        return sorted(causes, key=lambda x: x[1], reverse=True)
        
    def predict_effects(self, cause: str) -> List[Tuple[str, float]]:
        """Predict effects of cause"""
        
        return self.causal_graph.get(cause, [])
        
    def counterfactual(self, observation: str, intervention: str) -> str:
        """Generate counterfactual reasoning"""
        
        return f"If {intervention} had happened instead, then {observation} might have been different."

class ReasoningEngine:
    """Main reasoning engine integrating neural and symbolic reasoning"""
    
    def __init__(self, config):
        self.config = config
        self.neural_reasoner = NeuralReasoner(config)
        self.symbolic_reasoner = SymbolicReasoner()
        self.causal_reasoner = CausalReasoner()
        
        # Reasoning history
        self.reasoning_history = []
        
    def reason(self, context: Dict, question: str, 
               reasoning_type: str = "hybrid") -> Dict[str, Any]:
        """Main reasoning method"""
        
        logger.info(f"Starting {reasoning_type} reasoning", 
                   question=question,
                   context_keys=list(context.keys()))
        
        reasoning_steps = []
        
        # Neural reasoning
        if reasoning_type in ["hybrid", "neural"]:
            neural_answer, confidence = self.neural_reasoner.reason_about(
                context.get('text_context', []),
                question
            )
            
            reasoning_steps.append(ReasoningStep(
                step_type="neural_inference",
                content=neural_answer,
                confidence=confidence,
                evidence=context.get('text_context', [])[:3]
            ))
            
        # Symbolic reasoning
        if reasoning_type in ["hybrid", "symbolic"]:
            # Extract facts from context
            facts = self._extract_facts(context)
            
            # Add to symbolic reasoner
            for fact in facts:
                self.symbolic_reasoner.add_fact(fact)
                
            # Try to deduce answer
            success, proof = self.symbolic_reasoner.deduce(question)
            
            reasoning_steps.append(ReasoningStep(
                step_type="symbolic_deduction",
                content=f"Symbolic deduction {'succeeded' if success else 'failed'}",
                confidence=1.0 if success else 0.0,
                evidence=proof[:3]
            ))
            
        # Causal reasoning
        if reasoning_type in ["hybrid", "causal"]:
            causes = self.causal_reasoner.infer_causes(question)
            
            if causes:
                reasoning_steps.append(ReasoningStep(
                    step_type="causal_inference",
                    content=f"Found {len(causes)} potential causes",
                    confidence=min(1.0, len(causes) / 10.0),
                    evidence=[c[0] for c in causes[:3]]
                ))
                
        # Combine results
        final_answer = self._combine_reasoning(reasoning_steps)
        
        # Store in history
        self.reasoning_history.append({
            'question': question,
            'reasoning_steps': reasoning_steps,
            'final_answer': final_answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.reasoning_history) > 100:
            self.reasoning_history.pop(0)
            
        logger.info("Reasoning completed", 
                   answer_length=len(final_answer),
                   step_count=len(reasoning_steps))
        
        return {
            'answer': final_answer,
            'reasoning_steps': reasoning_steps,
            'confidence': self._calculate_confidence(reasoning_steps)
        }
        
    def _extract_facts(self, context: Dict) -> List[str]:
        """Extract facts from context"""
        facts = []
        
        # Extract from text
        for text in context.get('text_context', []):
            # Simple extraction - would use NLP in practice
            if len(text) < 100:
                facts.append(text)
                
        return facts
        
    def _combine_reasoning(self, steps: List[ReasoningStep]) -> str:
        """Combine multiple reasoning steps"""
        
        if not steps:
            return "Unable to reach conclusion."
            
        # Use highest confidence step
        best_step = max(steps, key=lambda s: s.confidence)
        
        return f"Based on {best_step.step_type}: {best_step.content}"
        
    def _calculate_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence"""
        
        if not steps:
            return 0.0
            
        confidences = [s.confidence for s in steps]
        return np.mean(confidences)
        
    def explain_reasoning(self, question: str) -> List[str]:
        """Generate explanation for reasoning"""
        
        # Find in history
        for entry in reversed(self.reasoning_history):
            if entry['question'] == question:
                explanation = [
                    f"Question: {question}",
                    f"Answer: {entry['final_answer']}",
                    "Reasoning process:"
                ]
                
                for step in entry['reasoning_steps']:
                    explanation.append(
                        f"  - [{step.step_type}] {step.content[:100]}... "
                        f"(confidence: {step.confidence:.2f})"
                    )
                    
                return explanation
                
        return ["No reasoning found for this question."]