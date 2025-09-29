from cyclic_compression_analysis import load_tracking_data
from pathlib import Path
import pandas as pd

# Load the Peak-Valley file
file_path = Path('0 WK/20250922 95-50 Reprint/Test Run 3 22-09-2025 3 31 51 pm/Cycle + DAQ- -6 mm to 0 mm at 1 Hz - Data Acquisition 1 - (Peak-Valley).csv')
print(f'Loading: {file_path.name}')
df = load_tracking_data(file_path)
if df is not None:
    print(f'Total data shape: {df.shape}')
    print(f'Cycles available: {df["Total Cycles"].unique()[:10]}...')
    print(f'Data points per cycle:')
    for cycle in [1, 2, 3, 4, 5]:
        cycle_data = df[df["Total Cycles"] == cycle]
        print(f'  Cycle {cycle}: {len(cycle_data)} points')
    print(f'\nFirst cycle data:')
    first_cycle = df[df["Total Cycles"] == 1]
    print(first_cycle)
else:
    print('Failed to load data')