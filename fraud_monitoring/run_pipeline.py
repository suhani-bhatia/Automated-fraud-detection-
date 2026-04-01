import os

print("Starting pipeline...")

print("Step 1: Preparing data...")
os.system("python prepare_data.py")

print("Step 2: Running monitoring...")
os.system("python monitor.py")

print("Pipeline execution complete.")