#!/usr/bin/env python3
"""
Quick script to check Prefect flow runs via API
"""
import requests
import json

try:
    # Prefect 3.x uses POST for queries
    response = requests.post(
        'http://localhost:4200/api/flow_runs/filter',
        json={"limit": 10, "offset": 0}
    )
    data = response.json()
    
    if isinstance(data, list) and data:
        print(f"\nFound {len(data)} flow run(s) in Prefect!")
        print()
        for i, run in enumerate(data, 1):
            print(f"{i}. Flow Run: {run.get('name', 'Unknown')}")
            print(f"   ID: {run.get('id', 'N/A')[:8]}...")
            state = run.get('state', {})
            print(f"   State: {state.get('type', 'Unknown') if isinstance(state, dict) else state}")
            print(f"   Created: {run.get('created', 'N/A')[:19]}")
            print()
    else:
        print("\nNo flow runs found yet.")
        print("Run the pipeline first:")
        print("  docker-compose exec pipeline python pipeline/prefect_flow.py")
        
except Exception as e:
    print(f"Error connecting to Prefect API: {e}")
    print("Make sure Prefect server is running: docker-compose ps")
