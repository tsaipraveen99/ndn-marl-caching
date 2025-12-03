#!/usr/bin/env python3
"""
Quick experiment runner for NDN MARL Caching simulation.

Usage:
    python run_experiment.py                    # Run default medium scenario
    python run_experiment.py --scenario large   # Run large network scenario
    python run_experiment.py --quick            # Quick test run
    python run_experiment.py --benchmark        # Run full benchmark comparison
"""

import sys
import os
import argparse
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(description='Run NDN MARL Caching Experiment')
    parser.add_argument('--scenario', type=str, default='medium',
                        choices=['minimal', 'medium', 'large', '500_nodes'],
                        help='Scenario to run (default: medium)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with reduced rounds')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run full benchmark comparison')
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
    
    if args.benchmark:
        # Run benchmark comparison
        print("Running benchmark comparison...")
        import subprocess
        cmd = ['python', 'benchmark.py', '--scenario', args.scenario]
        if args.quick:
            cmd.extend(['--rounds', '5'])
        subprocess.run(cmd)
    else:
        # Import and run simulation
        from main import create_network, run_simulation, setup_logging
        
        # Setup logging
        sim_logger, net_logger = setup_logging()
        
        # Create network
        G, users, producers, runtime = create_network(
            num_nodes=config.get('num_routers', 30),
            num_users=config.get('num_users', 30),
            num_producers=config.get('num_producers', 5),
            num_contents=config.get('content_count', 1200),
            cache_policy=args.policy,
            logger=net_logger
        )
        
        # Run simulation
        results = run_simulation(
            G=G,
            users=users,
            producers=producers,
            num_rounds=config.get('num_rounds', 40),
            num_requests=config.get('requests_per_round', 15),
            logger=sim_logger
        )
        
        # Print results
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        if results:
            hit_rate = results.get('cache_hit_rate', 0) * 100
            print(f"Cache Hit Rate: {hit_rate:.2f}%")
            print(f"Total Cache Hits: {results.get('total_cache_hits', 0)}")
            print(f"Total Requests: {results.get('total_requests', 0)}")
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

