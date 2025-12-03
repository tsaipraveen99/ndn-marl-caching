#!/usr/bin/env python3
"""
Quick experiment runner for NDN MARL Caching simulation.

Usage:
    python run_experiment.py                    # Run default medium scenario
    python run_experiment.py --scenario large   # Run large network scenario
    python run_experiment.py --quick            # Quick test run
"""

import sys
import os
import argparse
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import run_simulation


def main():
    parser = argparse.ArgumentParser(description='Run NDN MARL Caching Experiment')
    parser.add_argument('--scenario', type=str, default='medium',
                        choices=['minimal', 'medium', 'large', '500_nodes'],
                        help='Scenario to run (default: medium)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with reduced rounds')
    parser.add_argument('--policy', type=str, default='dqn',
                        choices=['dqn', 'lru', 'lfu', 'combined', 'probcache'],
                        help='Caching policy to use (default: dqn)')
    
    args = parser.parse_args()
    
    # Load config
    config_file = f'configs/scenario_{args.scenario}.json'
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Modify for quick run
    if args.quick:
        config['num_rounds'] = 5
        config['requests_per_round'] = 10
        print("Running quick test with 5 rounds...")
    
    print(f"\n{'='*60}")
    print(f"NDN MARL Caching Experiment")
    print(f"{'='*60}")
    print(f"Scenario: {args.scenario}")
    print(f"Routers: {config.get('num_routers', 30)}")
    print(f"Rounds: {config.get('num_rounds', 40)}")
    print(f"Policy: {args.policy}")
    print(f"{'='*60}\n")
    
    # Run simulation
    run_simulation(config, policy=args.policy)
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

