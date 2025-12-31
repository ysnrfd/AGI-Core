# agi_core/tools.py
"""
Tool execution system
"""

import inspect
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from utils.logger import logger

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Dict[str, Any]]
    returns: Dict[str, Any]
    category: str = "general"
    safety_level: int = 0  # 0-10, 10 = most dangerous
    
    def __post_init__(self):
        # Extract parameter schema from function if not provided
        if not self.parameters:
            self.parameters = self._extract_parameters()
            
    def _extract_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameter schema from function signature"""
        sig = inspect.signature(self.function)
        params = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'description': ''
            }
            
            params[name] = param_info
            
        return params
        
    def execute(self, **kwargs) -> Any:
        """Execute tool with given parameters"""
        try:
            logger.info(f"Executing tool: {self.name}", parameters=kwargs)
            
            # Validate parameters
            self._validate_parameters(kwargs)
            
            # Execute
            result = self.function(**kwargs)
            
            logger.info(f"Tool execution completed: {self.name}", 
                       result_type=type(result).__name__)
            
            return {
                'success': True,
                'result': result,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {self.name}", 
                        error=str(e),
                        parameters=kwargs)
            
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def _validate_parameters(self, kwargs: Dict[str, Any]):
        """Validate tool parameters"""
        
        for param_name, param_info in self.parameters.items():
            if param_info['required'] and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")
                
            if param_name in kwargs:
                # Type checking (simplified)
                expected_type = param_info['type']
                if expected_type != 'any':
                    # Simple type name checking
                    actual_type = type(kwargs[param_name]).__name__
                    if expected_type.lower() != actual_type.lower():
                        logger.warning(f"Type mismatch for {param_name}: "
                                     f"expected {expected_type}, got {actual_type}")

class ToolRegistry:
    """Registry for managing tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}
        
    def register(self, tool: Tool):
        """Register a tool"""
        
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
            
        self.tools[tool.name] = tool
        
        # Update categories
        if tool.category not in self.categories:
            self.categories[tool.category] = []
            
        if tool.name not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.name)
            
        logger.info(f"Registered tool: {tool.name}", 
                   category=tool.category,
                   parameters=list(tool.parameters.keys()))
        
    def register_function(self, func: Callable, 
                         name: Optional[str] = None,
                         description: Optional[str] = None,
                         category: str = "general",
                         safety_level: int = 0):
        """Register a function as a tool"""
        
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""
        
        tool = Tool(
            name=tool_name,
            description=tool_description,
            function=func,
            parameters={},
            returns={'type': 'any'},
            category=category,
            safety_level=safety_level
        )
        
        self.register(tool)
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
        
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools"""
        
        if category:
            return self.categories.get(category, [])
        else:
            return list(self.tools.keys())
            
    def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """Get tool schema for LLM consumption"""
        
        tool = self.get_tool(name)
        if not tool:
            return {}
            
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters,
            'returns': tool.returns,
            'category': tool.category,
            'safety_level': tool.safety_level
        }
        
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool"""
        
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                'success': False,
                'error': f"Tool not found: {tool_name}",
                'timestamp': datetime.now().isoformat()
            }
            
        return tool.execute(**kwargs)
        
    def search_tools(self, query: str, 
                    max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for tools by description"""
        
        results = []
        query_lower = query.lower()
        
        for tool in self.tools.values():
            score = 0
            
            # Match in name
            if query_lower in tool.name.lower():
                score += 3
                
            # Match in description
            if query_lower in tool.description.lower():
                score += 2
                
            # Match in category
            if query_lower in tool.category.lower():
                score += 1
                
            if score > 0:
                results.append({
                    'tool': tool.name,
                    'score': score,
                    'description': tool.description,
                    'category': tool.category
                })
                
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:max_results]

class ToolExecutor:
    """Execute tools with safety checks"""
    
    def __init__(self, registry: ToolRegistry, safety_checker):
        self.registry = registry
        self.safety_checker = safety_checker
        self.execution_history = []
        
    def execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a plan consisting of tool calls"""
        
        results = []
        
        for step in plan:
            tool_name = step.get('tool')
            parameters = step.get('parameters', {})
            
            if not tool_name:
                logger.warning("Plan step missing tool name", step=step)
                continue
                
            # Safety check
            safety_result = self.safety_checker.check_tool_use(
                tool_name, 
                parameters
            )
            
            if not safety_result['allowed']:
                logger.warning(f"Tool execution blocked by safety check: {tool_name}",
                             reason=safety_result.get('reason'))
                
                results.append({
                    'success': False,
                    'tool': tool_name,
                    'error': f"Safety check failed: {safety_result.get('reason')}",
                    'timestamp': datetime.now().isoformat()
                })
                continue
                
            # Execute tool
            result = self.registry.execute(tool_name, **parameters)
            result['tool'] = tool_name
            
            # Record in history
            self.execution_history.append({
                'tool': tool_name,
                'parameters': parameters,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            results.append(result)
            
            # Stop execution if step failed and plan requires success
            if not result['success'] and step.get('require_success', False):
                logger.warning(f"Plan execution stopped due to failed step: {tool_name}")
                break
                
        return results
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        if not self.execution_history:
            return {}
            
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history 
                        if h['result']['success'])
        
        # Tool usage frequency
        tool_counts = {}
        for h in self.execution_history:
            tool = h['tool']
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
        return {
            'total_executions': total,
            'success_rate': successful / total if total > 0 else 0,
            'tool_usage': tool_counts,
            'recent_executions': self.execution_history[-10:] if self.execution_history else []
        }

# Built-in tools
def web_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search the web for information"""
    # Placeholder - would integrate with actual search API
    logger.info(f"Searching web for: {query}", max_results=max_results)
    
    return [
        {'title': f"Result {i} for {query}", 
         'url': f"https://example.com/result{i}",
         'snippet': f"This is result {i} for the query: {query}"}
        for i in range(min(max_results, 5))
    ]

def execute_python(code: str) -> Dict[str, Any]:
    """Execute Python code"""
    try:
        # Safety check - don't execute dangerous operations
        dangerous_patterns = [
            'import os', 'import sys', '__import__', 'eval(',
            'exec(', 'open(', 'subprocess', 'rm ', 'del '
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                return {
                    'success': False,
                    'error': f"Dangerous operation detected: {pattern}",
                    'output': None
                }
                
        # Execute in isolated namespace
        namespace = {}
        exec(code, namespace)
        
        # Get output
        output = namespace.get('result', 'Code executed successfully')
        
        return {
            'success': True,
            'output': str(output),
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'output': None,
            'error': str(e)
        }

def file_read(path: str) -> str:
    """Read content from a file"""
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        raise Exception(f"Failed to read file {path}: {str(e)}")

def file_write(path: str, content: str) -> bool:
    """Write content to a file"""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        raise Exception(f"Failed to write file {path}: {str(e)}")

def calculate(expression: str) -> float:
    """Calculate mathematical expression"""
    try:
        # Safe evaluation
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains unsafe characters")
            
        result = eval(expression)
        return float(result)
    except Exception as e:
        raise Exception(f"Calculation failed: {str(e)}")

def get_current_time() -> str:
    """Get current date and time"""
    return datetime.now().isoformat()

# Initialize default tool registry
def create_default_registry() -> ToolRegistry:
    """Create registry with default tools"""
    
    registry = ToolRegistry()
    
    # Register default tools
    registry.register_function(
        web_search,
        name="web_search",
        description="Search the web for information",
        category="information",
        safety_level=1
    )
    
    registry.register_function(
        execute_python,
        name="execute_python",
        description="Execute Python code safely",
        category="computation",
        safety_level=5
    )
    
    registry.register_function(
        file_read,
        name="file_read",
        description="Read content from a file",
        category="file_io",
        safety_level=3
    )
    
    registry.register_function(
        file_write,
        name="file_write",
        description="Write content to a file",
        category="file_io",
        safety_level=6
    )
    
    registry.register_function(
        calculate,
        name="calculate",
        description="Calculate mathematical expression",
        category="computation",
        safety_level=0
    )
    
    registry.register_function(
        get_current_time,
        name="get_current_time",
        description="Get current date and time",
        category="utility",
        safety_level=0
    )
    
    return registry