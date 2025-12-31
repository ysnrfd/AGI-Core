# agi_core/main.py
"""
Main entry point for AGI Core
"""

import argparse
import json
from pathlib import Path

from agent import AGIAgent
from utils.logger import logger

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="AGI Core Framework")
    parser.add_argument("--goal", type=str, help="Initial goal for the agent")
    parser.add_argument("--iterations", type=int, default=100, 
                       help="Number of iterations to run")
    parser.add_argument("--config", type=str, 
                       help="Path to configuration file")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = config.from_yaml(config_path)
            
    logger.info("Starting AGI Core Framework", 
               config=config.to_dict())
    
    # Create agent
    agent = AGIAgent(name="DeepSeek-AGI")
    
    # Set initial goal
    if args.goal:
        agent.set_goal(args.goal)
    else:
        agent.set_goal("Explore and learn about the environment")
        
    # Run agent
    try:
        if args.interactive:
            run_interactive(agent)
        else:
            agent.run(max_iterations=args.iterations)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        
    finally:
        agent.shutdown()
        
    logger.info("AGI Core Framework execution complete")
    
def run_interactive(agent: AGIAgent):
    """Run agent in interactive mode"""
    
    print("\n" + "="*60)
    print("AGI Core Framework - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  run [iterations] - Run agent for N iterations")
    print("  goal [text]      - Set new goal")
    print("  status           - Show agent status")
    print("  memory           - Show memory stats")
    print("  tools            - List available tools")
    print("  quit             - Exit")
    print("="*60)
    
    while True:
        try:
            command = input("\nAGI> ").strip().split()
            
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == "quit" or cmd == "exit":
                break
                
            elif cmd == "run":
                iterations = int(command[1]) if len(command) > 1 else 10
                print(f"Running for {iterations} iterations...")
                agent.run(max_iterations=iterations)
                
            elif cmd == "goal":
                if len(command) > 1:
                    goal = " ".join(command[1:])
                    agent.set_goal(goal)
                    print(f"Goal set: {goal}")
                else:
                    print("Please provide a goal")
                    
            elif cmd == "status":
                status = agent.get_status()
                print(json.dumps(status, indent=2))
                
            elif cmd == "memory":
                status = agent.get_status()
                mem = status.get('memory_stats', {})
                print(f"Working memory: {mem.get('working', 0)} items")
                print(f"Long-term memory: {mem.get('long_term', 0)} items")
                print(f"Episodic memory: {mem.get('episodic', 0)} episodes")
                
            elif cmd == "tools":
                tools = agent.tool_registry.list_tools()
                print(f"Available tools ({len(tools)}):")
                for tool in tools:
                    print(f"  - {tool}")
                    
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
            
        except Exception as e:
            print(f"Error: {e}")
            
    print("Exiting interactive mode")

if __name__ == "__main__":
    main()