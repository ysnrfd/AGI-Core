# agi_core/utils/logger.py
"""
Logging utilities
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json

class AGILogger:
    """Custom logger for AGI system"""
    
    def __init__(self, name: str = "agi_core", log_dir: Path = Path("./logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = log_dir / f"agi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # JSON log for structured logging
        self.json_log_file = log_dir / f"structured_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
    def debug(self, msg: str, **kwargs):
        """Debug log with structured data"""
        self._log_structured("DEBUG", msg, **kwargs)
        self.logger.debug(msg)
        
    def info(self, msg: str, **kwargs):
        """Info log with structured data"""
        self._log_structured("INFO", msg, **kwargs)
        self.logger.info(msg)
        
    def warning(self, msg: str, **kwargs):
        """Warning log with structured data"""
        self._log_structured("WARNING", msg, **kwargs)
        self.logger.warning(msg)
        
    def error(self, msg: str, **kwargs):
        """Error log with structured data"""
        self._log_structured("ERROR", msg, **kwargs)
        self.logger.error(msg)
        
    def critical(self, msg: str, **kwargs):
        """Critical log with structured data"""
        self._log_structured("CRITICAL", msg, **kwargs)
        self.logger.critical(msg)
        
    def _log_structured(self, level: str, msg: str, **kwargs):
        """Log structured data to JSONL file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": msg,
            "data": kwargs
        }
        
        with open(self.json_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def log_agent_state(self, state: Dict[str, Any]):
        """Log agent state"""
        self.info("Agent state update", **state)

logger = AGILogger()