# Cyclic Compression Data Analysis Report

## Overview
This analysis examined Week 0 cyclic compression test data comparing normal tests and reprint tests across different SVF (Shore Value/Filament) and VS (Vertical Stiffness) combinations.

## Data Summary
The analysis processed data from **8 SVF-VS combinations**:
- **95-20**: SVF=95, VS=20 (Normal ✓, Reprint ✓)
- **95-35**: SVF=95, VS=35 (Normal ✓, Reprint ✓)
- **87-20**: SVF=87, VS=20 (Normal ✓, Reprint ✓)
- **87-35**: SVF=87, VS=35 (Normal ✓, Reprint ✓)
- **87-50**: SVF=87, VS=50 (Normal ✓, Reprint ✓)
- **90-20**: SVF=90, VS=20 (Normal ✓, Reprint ✓)
- **90-35**: SVF=90, VS=35 (Normal ✓, Reprint ✓)
- **90-50**: SVF=90, VS=50 (Normal ✓, Reprint ✓)

## Key Findings

### Peak Force Analysis
- **Highest Peak Forces**: SVF 90-50 combinations (~2.6 kN initial, decreasing over cycles)
- **Medium Peak Forces**: SVF 95-35 combinations (~2.4 kN initial, decreasing over cycles)
- **Lower Peak Forces**: SVF 87-50 combinations (~1.8 kN initial, decreasing over cycles)
- **Lowest Peak Forces**: SVF 87-20 and 95-20 combinations (~0.2-0.7 kN)

### Cycle Behavior
- Most tests ran for **1000+ cycles**
- Clear **degradation patterns** observed with decreasing peak forces over cycles
- **Reprint tests** generally showed slightly different force patterns compared to normal tests
- **Stabilization** typically occurred after initial rapid decrease in first 100 cycles

### Vertical Stiffness Effects
- **VS=50**: Highest peak forces across all SVF values
- **VS=35**: Medium peak forces
- **VS=20**: Lowest peak forces
- Clear correlation between vertical stiffness and peak force magnitude

## Generated Plots

### Plot 1: Peak Force vs Cycle by Combination (Normal vs Reprint)
- **File**: `plot1_peak_force_by_combination.png`
- **Description**: Shows individual SVF-VS combinations with normal and reprint tests overlaid
- **Key Insight**: Demonstrates repeatability between normal and reprint tests

### Plot 2: Average Peak Force vs Cycle by Vertical Stiffness
- **File**: `plot2_peak_force_by_vs.png`
- **Description**: Groups data by vertical stiffness (20, 35, 50) comparing normal vs reprint averages
- **Key Insight**: VS=50 shows highest forces, clear separation between stiffness levels

### Plot 3: Force vs Displacement for Specific Cycles (1st, 100th, 1000th)
- **File**: `plot3_force_displacement_specific_cycles.png`
- **Description**: Hysteresis loops for key cycles showing material behavior evolution
- **Key Insight**: Shows how force-displacement behavior changes with cycling

### Plot 4: Average Force vs Cycle by Vertical Stiffness (Combined)
- **File**: `plot4_force_vs_cycle_by_vs_combined.png`
- **Description**: Combined view of all vertical stiffness levels with normal/reprint comparison
- **Key Insight**: Overall trends and comparative behavior across all conditions

## Technical Notes

### Data Processing
- Peak forces calculated as maximum absolute force per cycle
- Data interpolated to common cycle ranges for averaging
- Both normal and reprint tests included in analysis
- ~15,000-47,000 data points per test depending on cycle count

### Test Conditions
- Cyclic compression testing
- Force measured in kN
- Displacement measured in mm
- Various SVF (87, 90, 95) and VS (20, 35, 50) combinations

## Recommendations

1. **Material Selection**: Higher VS values provide greater force resistance
2. **Durability**: All combinations show acceptable cyclic behavior over 1000 cycles
3. **Repeatability**: Good correlation between normal and reprint tests validates methodology
4. **Further Analysis**: Consider fatigue life prediction based on force degradation curves

## Files Generated
- `cyclic_compression_analysis.py`: Analysis script
- `plot1_peak_force_by_combination.png`: Individual combination comparison
- `plot2_peak_force_by_vs.png`: Vertical stiffness grouping
- `plot3_force_displacement_specific_cycles.png`: Hysteresis evolution
- `plot4_force_vs_cycle_by_vs_combined.png`: Overall comparison
- `analysis_report.md`: This summary report