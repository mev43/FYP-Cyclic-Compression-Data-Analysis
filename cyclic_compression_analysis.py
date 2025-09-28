import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def get_filament_name(filament_number):
    """
    Convert filament number to proper TPU name
    """
    filament_map = {87: 'TPU87A', 90: 'TPU90A', 95: 'TPU95A'}
    return filament_map.get(filament_number, f'TPU{filament_number}A')

def create_output_directories():
    """
    Create output directories for organizing plots
    """
    base_dir = Path('Test 1 analysis_results')
    dirs = [
        base_dir / 'Test 1 vertical_stiffness_comparison',
        base_dir / 'Test 1 force_displacement_cycles',
        base_dir / 'Test 1 force_displacement_by_vs_groups',
        base_dir / 'Test 1 by_filament_type',
        base_dir / 'Test 1 by_svf_percentage'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def extract_parameters_from_filename(folder_name):
    """
    Extract Filament type and SVF (Solid Volume Fraction) from folder name
    Examples: 
    - '20250825 95-35 3DEA1' -> Filament=95, SVF=35
    - '20250909 87-20 REPRINT' -> Filament=87, SVF=20
    """
    parts = folder_name.split()
    if len(parts) >= 2:
        # Extract Filament-SVF pattern (e.g., '95-35', '87-20')
        filament_svf = parts[1]
        if '-' in filament_svf:
            filament, svf = filament_svf.split('-')
            return int(filament), int(svf)
    return None, None

def calculate_vertical_stiffness(df):
    """
    Calculate vertical stiffness from the first cycle force-displacement data
    Vertical stiffness = slope of force vs displacement during loading phase
    """
    try:
        # Get first cycle data
        first_cycle_data = df[df['Total Cycles'] == 1].copy()
        
        if first_cycle_data.empty:
            return None
        
        # Get force and displacement data
        force = first_cycle_data['Force(Linear:Load) (kN)'].values
        displacement = first_cycle_data['Displacement(Linear:Digital Position) (mm)'].values
        
        if len(force) < 10 or len(displacement) < 10:  # Need sufficient data points
            return None
        
        # Sort data by displacement to ensure proper ordering
        sort_indices = np.argsort(displacement)
        displacement_sorted = displacement[sort_indices]
        force_sorted = force[sort_indices]
        
        # Find the main loading region (where force increases with displacement)
        # Take the portion from minimum displacement to maximum force
        min_disp_idx = 0
        max_force_idx = np.argmax(np.abs(force_sorted))
        
        # Ensure we have a reasonable range
        if max_force_idx < len(force_sorted) * 0.2:  # If max force is too early
            max_force_idx = int(len(force_sorted) * 0.6)  # Use first 60% of data
        
        # Extract loading phase data
        disp_loading = displacement_sorted[min_disp_idx:max_force_idx]
        force_loading = force_sorted[min_disp_idx:max_force_idx]
        
        # Remove any data points with zero or very small displacement changes
        if len(disp_loading) >= 5:
            disp_range = np.max(disp_loading) - np.min(disp_loading)
            
            if disp_range > 0.001:  # Minimum displacement range threshold
                # Calculate stiffness using linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(disp_loading, force_loading)
                
                # Only return stiffness if the correlation is reasonable (R² > 0.3)
                # and slope makes physical sense (positive for compression)
                if r_value**2 > 0.3 and abs(slope) > 0.001:
                    # Return absolute value of slope (stiffness) in N/mm
                    # Convert from kN/mm to N/mm by multiplying by 1000
                    vertical_stiffness = abs(slope) * 1000
                    return vertical_stiffness
        
        return None
        
    except Exception as e:
        print(f"Error calculating vertical stiffness: {e}")
        return None

def find_peak_force_per_cycle(df):
    """
    Find peak force for each cycle in the tracking data
    """
    peak_forces = []
    cycle_numbers = []
    
    # Group by cycle number
    cycles = df['Total Cycles'].unique()
    
    for cycle in cycles:
        cycle_data = df[df['Total Cycles'] == cycle]
        if not cycle_data.empty:
            # Find peak force (maximum absolute value) in this cycle
            forces = cycle_data['Force(Linear:Load) (kN)'].values
            peak_force = np.max(np.abs(forces))  # Peak absolute force
            peak_forces.append(peak_force)
            cycle_numbers.append(cycle)
    
    return np.array(cycle_numbers), np.array(peak_forces)

def load_tracking_data(file_path):
    """
    Load tracking CSV file and return DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_week0_data(base_path):
    """
    Analyze Week 0 data including normal and reprint folders
    Calculate vertical stiffness from first cycle data
    """
    week0_path = Path(base_path) / "0 WK"
    
    # Dictionary to store data organized by Filament-SVF-VS combination
    data_by_combination = {}
    
    # Find all folders in week 0
    for folder in week0_path.iterdir():
        if folder.is_dir():
            filament, svf = extract_parameters_from_filename(folder.name)
            if filament is not None and svf is not None:
                # Determine if this is a reprint or normal test
                test_type = "reprint" if "REPRINT" in folder.name.upper() else "normal"
                
                # Look for tracking CSV file
                tracking_file = folder / "Test1" / "Test1.steps.tracking.csv"
                if tracking_file.exists():
                    df = load_tracking_data(tracking_file)
                    if df is not None:
                        cycles, peak_forces = find_peak_force_per_cycle(df)
                        
                        # Calculate vertical stiffness from first cycle
                        vs_calculated = calculate_vertical_stiffness(df)
                        
                        if vs_calculated is not None:
                            # Round to reasonable precision for grouping (10 N/mm increments)
                            vs_rounded = round(vs_calculated / 10) * 10
                            combination_key = f"Filament{filament}_SVF{svf}_VS{vs_rounded}"
                            
                            if combination_key not in data_by_combination:
                                data_by_combination[combination_key] = {
                                    'filament': filament, 'svf': svf, 'vs': vs_rounded, 'normal': {}, 'reprint': {}
                                }
                            
                            data_by_combination[combination_key][test_type] = {
                                'cycles': cycles,
                                'peak_forces': peak_forces,
                                'full_data': df,
                                'folder_name': folder.name,
                                'calculated_vs': vs_calculated
                            }
                        else:
                            print(f"Could not calculate vertical stiffness for {folder.name}")
    
    return data_by_combination

def group_similar_vs(data_by_combination, tolerance=40):
    """
    Group combinations with vertical stiffnesses within tolerance (N/mm) of each other
    """
    vs_values = [combo_data['vs'] for combo_data in data_by_combination.values()]
    vs_values = sorted(set(vs_values))  # Get unique VS values, sorted
    
    vs_groups = []
    used_vs = set()
    
    for vs in vs_values:
        if vs in used_vs:
            continue
            
        # Find all VS values within tolerance
        group = [vs]
        used_vs.add(vs)
        
        for other_vs in vs_values:
            if other_vs != vs and other_vs not in used_vs:
                if abs(vs - other_vs) <= tolerance:
                    group.append(other_vs)
                    used_vs.add(other_vs)
        
        vs_groups.append(group)
    
    return vs_groups

def plot_vs_groups(data_by_combination, vs_groups):
    """
    Plot average peak force vs cycle for grouped vertical stiffnesses (within 40 N/mm)
    Only shows combinations with different filament OR different SVF
    """
    for group_idx, vs_group in enumerate(vs_groups):
                # Check if this group has different combinations (different filament OR SVF)
        combinations_in_group = []
        for vs in vs_group:
            for combination_key, combo_data in data_by_combination.items():
                if combo_data['vs'] == vs:
                    filament, svf = combo_data['filament'], combo_data['svf']
                    combinations_in_group.append((filament, svf, vs))
        
        # Check if we have different combinations (not just same filament-SVF with different VS)
        unique_combinations = set((f, s) for f, s, v in combinations_in_group)
        if len(unique_combinations) <= 1:
            print(f"Skipping VS Group {group_idx+1} - only contains identical filament-SVF combinations")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(vs_group)))
        
        # Sort VS values in group for consistent plotting
        vs_group = sorted(vs_group)
        
        for idx, vs in enumerate(vs_group):
            color = colors[idx]
            
            # Collect all test data for this VS
            all_forces_data = []
            combinations_info = []
            
            for combination_key, combo_data in data_by_combination.items():
                if combo_data['vs'] == vs:
                    filament, svf = combo_data['filament'], combo_data['svf']
                    filament_name = get_filament_name(filament)
                    combinations_info.append(f"{filament_name}-SVF{svf}%")
                    
                    # Add normal test data
                    if 'normal' in combo_data and combo_data['normal']:
                        cycles = combo_data['normal']['cycles']
                        forces = combo_data['normal']['peak_forces']
                        all_forces_data.append((cycles, forces))
                    
                    # Add reprint test data
                    if 'reprint' in combo_data and combo_data['reprint']:
                        cycles = combo_data['reprint']['cycles']
                        forces = combo_data['reprint']['peak_forces']
                        all_forces_data.append((cycles, forces))
            
            if all_forces_data:
                # Find maximum cycle count
                max_cycles = max(len(forces) for cycles, forces in all_forces_data)
                common_cycles = np.arange(1, max_cycles + 1)
                
                # Interpolate all tests to common cycle range
                interpolated_forces = []
                for cycles, forces in all_forces_data:
                    interp_forces = np.interp(common_cycles, cycles, forces)
                    interpolated_forces.append(interp_forces)
                
                # Calculate average and standard deviation
                avg_forces = np.mean(interpolated_forces, axis=0)
                std_forces = np.std(interpolated_forces, axis=0) if len(interpolated_forces) > 1 else np.zeros_like(avg_forces)
                
                # Create label with combinations info
                combo_info = ", ".join(set(combinations_info))  # Remove duplicates
                label = f'VS = {vs} N/mm ({combo_info})'
                
                # Plot average with error bands
                ax.plot(common_cycles, avg_forces, '-', color=color, 
                       label=label, linewidth=2, alpha=0.8)
                
                # Add standard deviation bands if multiple tests
                if len(interpolated_forces) > 1:
                    ax.fill_between(common_cycles, 
                                  avg_forces - std_forces/2, 
                                  avg_forces + std_forces/2, 
                                  color=color, alpha=0.2)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Average Peak Force (kN)')
        
        # Create title with VS range and combination info
        vs_range = f"{min(vs_group)}-{max(vs_group)}" if len(vs_group) > 1 else str(vs_group[0])
        combo_summary = ", ".join([f"{get_filament_name(f)}-SVF {s}%" for f, s in unique_combinations])
        ax.set_title(f'Test 1 Peak Force vs Cycle (Averaged over two sets) - Similar Vertical Stiffnesses\nVS Group: {vs_range} N/mm\nCombinations: {combo_summary}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(f'Test 1 analysis_results/Test 1 vertical_stiffness_comparison/Test 1 VS_Group_{group_idx+1}_{vs_range}_Nmm_different_combinations.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"VS Group {group_idx+1} ({vs_range} N/mm) with different combinations saved as: {output_path}")

def plot_specific_cycles_force_displacement(data_by_combination, cycles_of_interest=[1, 100, 1000]):
    """
    Plot 3: Individual force vs displacement plots for each combination, showing all cycles on one plot
    Only plot averages with normalized displacement (0 to -6mm range)
    """
    def average_hysteresis_loops(displacements_list, forces_list):
        """Helper function to properly average hysteresis loops by preserving temporal order"""
        if not displacements_list or not forces_list:
            return None, None
        
        # Process each dataset to clean and normalize
        processed_data = []
        for disp, force in zip(displacements_list, forces_list):
            # Remove any NaN values
            valid_mask = ~(np.isnan(disp) | np.isnan(force))
            disp_clean = disp[valid_mask]
            force_clean = force[valid_mask]
            
            if len(disp_clean) < 10:  # Need sufficient points for a hysteresis loop
                continue
            
            # Normalize displacement to start at 0 and go negative (compression)
            # Find the minimum displacement (most compressed point)
            min_disp = np.min(disp_clean)
            max_disp = np.max(disp_clean)
            
            # Shift displacement so the maximum (least compressed) is at 0
            # and minimum (most compressed) is around -6mm
            disp_normalized = disp_clean - max_disp
            
            # Scale if needed to get approximately -6mm range
            disp_range = abs(min_disp - max_disp)
            if disp_range > 0.1:  # Avoid division by zero
                target_range = 6.0  # Target 6mm compression range
                scale_factor = target_range / disp_range
                disp_normalized = disp_normalized * scale_factor
            
            processed_data.append((disp_normalized, force_clean))
        
        if len(processed_data) < 1:
            return None, None
        
        # If we only have one dataset, return it as is
        if len(processed_data) == 1:
            return processed_data[0]
        
        # For multiple datasets, use time-based averaging
        # Find the dataset with the most points to use as reference
        ref_idx = np.argmax([len(data[0]) for data in processed_data])
        ref_disp, ref_force = processed_data[ref_idx]
        
        # For other datasets, interpolate based on temporal progression
        all_forces = [ref_force]
        
        for i, (disp, force) in enumerate(processed_data):
            if i == ref_idx:
                continue
                
            # Create normalized time parameter for both datasets
            ref_time = np.linspace(0, 1, len(ref_force))
            data_time = np.linspace(0, 1, len(force))
            
            # Interpolate this dataset's force to match reference timing
            interp_force = np.interp(ref_time, data_time, force)
            all_forces.append(interp_force)
        
        # Average all forces at corresponding time points
        avg_force = np.mean(all_forces, axis=0)
        
        return ref_disp, avg_force
    
    # Define colors for each cycle
    cycle_colors = {1: 'blue', 100: 'red', 1000: 'green'}
    
    # Create individual plots for each combination with all cycles on one plot
    for combination_key, combo_data in data_by_combination.items():
        filament, svf, vs = combo_data['filament'], combo_data['svf'], combo_data['vs']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot each cycle on the same axes
        for target_cycle in cycles_of_interest:
            # Collect data for this cycle from both normal and reprint tests
            displacements_list = []
            forces_list = []
            
            # Get normal test data
            if 'normal' in combo_data and combo_data['normal'] and 'full_data' in combo_data['normal']:
                df_normal = combo_data['normal']['full_data']
                cycle_data = df_normal[df_normal['Total Cycles'] == target_cycle]
                if not cycle_data.empty:
                    displacement = cycle_data['Displacement(Linear:Digital Position) (mm)'].values
                    force = cycle_data['Force(Linear:Load) (kN)'].values
                    if len(displacement) > 5 and len(force) > 5:  # Ensure sufficient data
                        displacements_list.append(displacement)
                        forces_list.append(force)
            
            # Get reprint test data
            if 'reprint' in combo_data and combo_data['reprint'] and 'full_data' in combo_data['reprint']:
                df_reprint = combo_data['reprint']['full_data']
                cycle_data = df_reprint[df_reprint['Total Cycles'] == target_cycle]
                if not cycle_data.empty:
                    displacement = cycle_data['Displacement(Linear:Digital Position) (mm)'].values
                    force = cycle_data['Force(Linear:Load) (kN)'].values
                    if len(displacement) > 5 and len(force) > 5:  # Ensure sufficient data
                        displacements_list.append(displacement)
                        forces_list.append(force)
            
            # Calculate and plot average force-displacement curve for this cycle
            if displacements_list and forces_list:
                avg_disp, avg_force = average_hysteresis_loops(displacements_list, forces_list)
                
                if avg_disp is not None and avg_force is not None:
                    # Only plot averaged curve (no individual curves)
                    ax.plot(avg_disp, avg_force, 
                           color=cycle_colors.get(target_cycle, 'black'), 
                           linewidth=3, alpha=0.9, 
                           label=f'Cycle {target_cycle}')
        
        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Force (kN)')
        filament_name = get_filament_name(filament)
        ax.set_title(f'Test 1 Force vs Displacement (Averaged over two sets) - {filament_name}, SVF {svf}%\n(Cycles 1, 100, 1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show compression (negative values)
        ax.set_xlim(-7, 1)  # Show from -7mm to +1mm for better visualization
        
        plt.tight_layout()
        
        # Save individual combination plot with Filament and SVF identification
        filament_name_file = get_filament_name(filament)
        output_path = Path(f'Test 1 analysis_results/Test 1 force_displacement_cycles/Test 1 {filament_name_file}_SVF_{svf}_percent_force_displacement_cycles.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Force-displacement plot for {filament_name}, SVF {svf}% saved as: {output_path}")

def plot_combined_average_by_vs(data_by_combination):
    """
    Plot 2b: Overall combined average peak force vs cycle for similar vertical stiffnesses
    (All tests combined - normal and reprint averaged together)
    """
    # Group by vertical stiffness
    vs_groups = {}
    for combination_key, combo_data in data_by_combination.items():
        vs = combo_data['vs']
        if vs not in vs_groups:
            vs_groups[vs] = []
        vs_groups[vs].append(combo_data)
    
    # Sort by VS for consistent ordering
    vs_groups = dict(sorted(vs_groups.items()))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(vs_groups)))
    
    for idx, (vs, combinations) in enumerate(vs_groups.items()):
        color = colors[idx]
        
        # Collect all test data (both normal and reprint)
        all_forces_data = []
        max_cycles = 0
        
        for combo_data in combinations:
            # Add normal test data
            if 'normal' in combo_data and combo_data['normal']:
                cycles = combo_data['normal']['cycles']
                forces = combo_data['normal']['peak_forces']
                all_forces_data.append((cycles, forces))
                max_cycles = max(max_cycles, len(forces))
            
            # Add reprint test data
            if 'reprint' in combo_data and combo_data['reprint']:
                cycles = combo_data['reprint']['cycles']
                forces = combo_data['reprint']['peak_forces']
                all_forces_data.append((cycles, forces))
                max_cycles = max(max_cycles, len(forces))
        
        if max_cycles > 0 and all_forces_data:
            common_cycles = np.arange(1, max_cycles + 1)
            
            # Interpolate all tests to common cycle range
            interpolated_forces = []
            for cycles, forces in all_forces_data:
                interp_forces = np.interp(common_cycles, cycles, forces)
                interpolated_forces.append(interp_forces)
            
            # Calculate overall average
            avg_forces = np.mean(interpolated_forces, axis=0)
            std_forces = np.std(interpolated_forces, axis=0)
            
            # Plot average with error bands
            ax.plot(common_cycles, avg_forces, '-', color=color, 
                   label=f'VS = {vs} N/mm (n={len(all_forces_data)})', 
                   linewidth=3, alpha=0.8)
            
            # Add standard deviation bands
            ax.fill_between(common_cycles, 
                          avg_forces - std_forces/2, 
                          avg_forces + std_forces/2, 
                          color=color, alpha=0.2)
    
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Average Peak Force (kN)')
    ax.set_title('Test 1 Combined Average Peak Force vs Cycle by Vertical Stiffness\n(All Normal & Reprint Tests Averaged)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot in appropriate folder
    output_path = Path('Test 1 analysis_results/Test 1 vertical_stiffness_comparison/Test 1 combined_average_by_vs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined VS average plot saved as: {output_path}")

def plot_by_filament_type(data_by_combination):
    """
    Plot 6: Average peak force vs cycle grouped by filament type (same SVF, different filaments)
    """
    # Group by SVF percentage
    svf_groups = {}
    for combination_key, combo_data in data_by_combination.items():
        svf = combo_data['svf']
        if svf not in svf_groups:
            svf_groups[svf] = {}
        
        filament = combo_data['filament']
        if filament not in svf_groups[svf]:
            svf_groups[svf][filament] = {'normal': [], 'reprint': []}
        
        if 'normal' in combo_data and combo_data['normal']:
            svf_groups[svf][filament]['normal'].append(combo_data['normal'])
        if 'reprint' in combo_data and combo_data['reprint']:
            svf_groups[svf][filament]['reprint'].append(combo_data['reprint'])
    
    # Create plots for each SVF percentage
    for svf, filament_data in svf_groups.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(filament_data)))
        
        for idx, (filament, test_data) in enumerate(filament_data.items()):
            color = colors[idx]
            
            # Combine normal and reprint data for this filament
            all_cycles = []
            all_peak_forces = []
            
            # Add normal test data
            for normal_test in test_data['normal']:
                cycles = normal_test.get('cycles', [])
                peak_forces = normal_test.get('peak_forces', [])
                all_cycles.extend(cycles)
                all_peak_forces.extend(peak_forces)
            
            # Add reprint test data
            for reprint_test in test_data['reprint']:
                cycles = reprint_test.get('cycles', [])
                peak_forces = reprint_test.get('peak_forces', [])
                all_cycles.extend(cycles)
                all_peak_forces.extend(peak_forces)
            
            if all_cycles:
                # Create bins for averaging
                max_cycle = max(all_cycles)
                bin_edges = np.linspace(1, max_cycle, min(100, max_cycle))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                binned_forces = []
                for i in range(len(bin_edges) - 1):
                    mask = (np.array(all_cycles) >= bin_edges[i]) & (np.array(all_cycles) < bin_edges[i+1])
                    if np.any(mask):
                        binned_forces.append(np.mean(np.array(all_peak_forces)[mask]))
                    else:
                        binned_forces.append(np.nan)
                
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(binned_forces)
                if np.any(valid_mask):
                    filament_name = get_filament_name(filament)
                    ax.plot(bin_centers[valid_mask], np.array(binned_forces)[valid_mask], 
                           '-', color=color, label=f'{filament_name}', 
                           linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Average Peak Force (kN)')
        ax.set_title(f'Test 1 Average Peak Force vs Cycle - SVF {svf}%\n(Comparison across Filament Types)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add SVF annotation
        ax.text(0.02, 0.98, f'SVF {svf}%', transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                verticalalignment='top', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(f'Test 1 analysis_results/Test 1 by_filament_type/Test 1 SVF_{svf}_percent_filament_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Filament comparison plot for SVF {svf}% saved as: {output_path}")

def plot_by_svf_percentage(data_by_combination):
    """
    Plot 7: Average peak force vs cycle grouped by SVF percentage (same filament, different SVFs)
    """
    # Group by filament type
    filament_groups = {}
    for combination_key, combo_data in data_by_combination.items():
        filament = combo_data['filament']
        if filament not in filament_groups:
            filament_groups[filament] = {}
        
        svf = combo_data['svf']
        if svf not in filament_groups[filament]:
            filament_groups[filament][svf] = {'normal': [], 'reprint': []}
        
        if 'normal' in combo_data and combo_data['normal']:
            filament_groups[filament][svf]['normal'].append(combo_data['normal'])
        if 'reprint' in combo_data and combo_data['reprint']:
            filament_groups[filament][svf]['reprint'].append(combo_data['reprint'])
    
    # Create plots for each filament type
    for filament, svf_data in filament_groups.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(svf_data)))
        
        for idx, (svf, test_data) in enumerate(svf_data.items()):
            color = colors[idx]
            
            # Combine normal and reprint data for this SVF
            all_cycles = []
            all_peak_forces = []
            
            # Add normal test data
            for normal_test in test_data['normal']:
                cycles = normal_test.get('cycles', [])
                peak_forces = normal_test.get('peak_forces', [])
                all_cycles.extend(cycles)
                all_peak_forces.extend(peak_forces)
            
            # Add reprint test data
            for reprint_test in test_data['reprint']:
                cycles = reprint_test.get('cycles', [])
                peak_forces = reprint_test.get('peak_forces', [])
                all_cycles.extend(cycles)
                all_peak_forces.extend(peak_forces)
            
            if all_cycles:
                # Create bins for averaging
                max_cycle = max(all_cycles)
                bin_edges = np.linspace(1, max_cycle, min(100, max_cycle))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                binned_forces = []
                for i in range(len(bin_edges) - 1):
                    mask = (np.array(all_cycles) >= bin_edges[i]) & (np.array(all_cycles) < bin_edges[i+1])
                    if np.any(mask):
                        binned_forces.append(np.mean(np.array(all_peak_forces)[mask]))
                    else:
                        binned_forces.append(np.nan)
                
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(binned_forces)
                if np.any(valid_mask):
                    ax.plot(bin_centers[valid_mask], np.array(binned_forces)[valid_mask], 
                           '-', color=color, label=f'SVF {svf}%', 
                           linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Average Peak Force (kN)')
        filament_name = get_filament_name(filament)
        ax.set_title(f'Test 1 Peak Force vs Cycle (Averaged over two sets) - {filament_name}\n(Comparison across SVF Percentages)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add filament annotation
        ax.text(0.02, 0.98, f'{filament_name}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9),
                verticalalignment='top', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filament_name_file = get_filament_name(filament)
        output_path = Path(f'Test 1 analysis_results/Test 1 by_svf_percentage/Test 1 {filament_name_file}_SVF_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SVF comparison plot for {filament_name} saved as: {output_path}")

def plot_specific_cycles_by_vs_groups(data_by_combination, vs_groups, target_cycles=[1, 100, 1000]):
    """
    Plot force vs displacement for specific cycles (1st, 100th, 1000th) 
    with separate plots for each VS group and each cycle.
    Each VS group gets its own folder with plots showing individual combination curves.
    """
    def average_hysteresis_loops(displacements_list, forces_list):
        """Helper function to properly average hysteresis loops by preserving temporal order"""
        if not displacements_list or not forces_list:
            return None, None
        
        # Process each dataset to clean and normalize
        processed_data = []
        for disp, force in zip(displacements_list, forces_list):
            # Remove any NaN values
            valid_mask = ~(np.isnan(disp) | np.isnan(force))
            disp_clean = disp[valid_mask]
            force_clean = force[valid_mask]
            
            if len(disp_clean) < 10:  # Need sufficient points for a hysteresis loop
                continue
            
            # Normalize displacement to start at 0 and go negative (compression)
            # Find the minimum displacement (most compressed point)
            min_disp = np.min(disp_clean)
            max_disp = np.max(disp_clean)
            
            # Shift displacement so the maximum (least compressed) is at 0
            # and minimum (most compressed) is around -6mm
            disp_normalized = disp_clean - max_disp
            
            # Scale if needed to get approximately -6mm range
            disp_range = abs(min_disp - max_disp)
            if disp_range > 0.1:  # Avoid division by zero
                target_range = 6.0  # Target 6mm compression range
                scale_factor = target_range / disp_range
                disp_normalized = disp_normalized * scale_factor
            
            processed_data.append((disp_normalized, force_clean))
        
        if len(processed_data) < 1:
            return None, None
        
        # If we only have one dataset, return it as is
        if len(processed_data) == 1:
            return processed_data[0]
        
        # For multiple datasets, use time-based averaging
        # Find the dataset with the most points to use as reference
        ref_idx = np.argmax([len(data[0]) for data in processed_data])
        ref_disp, ref_force = processed_data[ref_idx]
        
        # For other datasets, interpolate based on temporal progression
        all_forces = [ref_force]
        
        for i, (disp, force) in enumerate(processed_data):
            if i == ref_idx:
                continue
                
            # Create normalized time parameter for both datasets
            ref_time = np.linspace(0, 1, len(ref_force))
            data_time = np.linspace(0, 1, len(force))
            
            # Interpolate this dataset's force to match reference timing
            interp_force = np.interp(ref_time, data_time, force)
            all_forces.append(interp_force)
        
        # Average all forces at corresponding time points
        avg_force = np.mean(all_forces, axis=0)
        
        return ref_disp, avg_force
    
    for group_idx, vs_group in enumerate(vs_groups):
        # Check if this group has different combinations (different filament OR SVF)
        combinations_in_group = []
        for vs in vs_group:
            for combination_key, combo_data in data_by_combination.items():
                if combo_data['vs'] == vs:
                    filament, svf = combo_data['filament'], combo_data['svf']
                    combinations_in_group.append((filament, svf, vs))
        
        # Check if we have different combinations (not just same filament-SVF with different VS)
        unique_combinations = set((f, s) for f, s, v in combinations_in_group)
        if len(unique_combinations) <= 1:
            print(f"Skipping VS Group {group_idx+1} - only contains identical filament-SVF combinations")
            continue
        
        # Create VS group folder
        vs_range = f"{min(vs_group)}-{max(vs_group)}" if len(vs_group) > 1 else str(vs_group[0])
        group_folder = Path(f'Test 1 analysis_results/Test 1 force_displacement_by_vs_groups/Test 1 VS_Group_{group_idx+1}_{vs_range}_Nmm')
        group_folder.mkdir(parents=True, exist_ok=True)
        
        for target_cycle in target_cycles:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create colors for each unique combination in this VS group
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_combinations)))
            combination_colors = {combo: colors[i] for i, combo in enumerate(sorted(unique_combinations))}
            
            plot_data_exists = False
            # Plot each unique combination separately (don't average across different combinations)
            for vs in vs_group:
                for combination_key, combo_data in data_by_combination.items():
                    if combo_data['vs'] == vs:
                        filament, svf = combo_data['filament'], combo_data['svf']
                        combo_tuple = (filament, svf)
                        
                        # Collect force-displacement data for this specific combination and target cycle
                        all_displacements = []
                        all_forces = []
                        
                        # Get force-displacement data for target cycle from normal test
                        if 'normal' in combo_data and combo_data['normal'] and 'full_data' in combo_data['normal']:
                            df_normal = combo_data['normal']['full_data']
                            cycle_data = df_normal[df_normal['Total Cycles'] == target_cycle]
                            if not cycle_data.empty:
                                displacement = cycle_data['Displacement(Linear:Digital Position) (mm)'].values
                                force = cycle_data['Force(Linear:Load) (kN)'].values
                                if len(displacement) > 5 and len(force) > 5:  # Ensure sufficient data
                                    all_displacements.append(displacement)
                                    all_forces.append(force)
                        
                        # Get force-displacement data for target cycle from reprint test
                        if 'reprint' in combo_data and combo_data['reprint'] and 'full_data' in combo_data['reprint']:
                            df_reprint = combo_data['reprint']['full_data']
                            cycle_data = df_reprint[df_reprint['Total Cycles'] == target_cycle]
                            if not cycle_data.empty:
                                displacement = cycle_data['Displacement(Linear:Digital Position) (mm)'].values
                                force = cycle_data['Force(Linear:Load) (kN)'].values
                                if len(displacement) > 5 and len(force) > 5:  # Ensure sufficient data
                                    all_displacements.append(displacement)
                                    all_forces.append(force)
                        
                        # Calculate and plot average force-displacement curve for this specific combination
                        if all_displacements and all_forces:
                            avg_disp, avg_force = average_hysteresis_loops(all_displacements, all_forces)
                            
                            if avg_disp is not None and avg_force is not None:
                                plot_data_exists = True
                                
                                color = combination_colors[combo_tuple]
                                filament_name = get_filament_name(filament)
                                label = f'{filament_name}-SVF {svf}% (VS={vs} N/mm)'
                                
                                # Plot averaged curve for this specific combination
                                ax.plot(avg_disp, avg_force, 
                                       color=color, 
                                       linewidth=3, alpha=0.8, 
                                       label=label)
            
            # Save plot after processing all combinations for this cycle
            if plot_data_exists:
                ax.set_xlabel('Displacement (mm)')
                ax.set_ylabel('Force (kN)')
                combo_summary = ", ".join([f"{get_filament_name(f)}-SVF {s}%" for f, s in sorted(unique_combinations)])
                ax.set_title(f'Test 1 Force vs Displacement (Averaged over two sets) - Cycle {target_cycle}\nVS Group: {vs_range} N/mm\nCombinations: {combo_summary}')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Set x-axis to show compression (negative values)
                ax.set_xlim(-7, 1)  # Show from -7mm to +1mm for better visualization
                
                plt.tight_layout()
                
                # Save plot in the VS group folder
                output_path = group_folder / f'Test 1 Cycle_{target_cycle}_force_displacement.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Cycle {target_cycle} force-displacement for VS Group {group_idx+1} ({vs_range} N/mm) saved as: {output_path}")
            else:
                plt.close()
                print(f"No data found for cycle {target_cycle} in VS Group {group_idx+1} ({vs_range} N/mm)")

def main():
    """
    Main analysis function
    """
    # Set the base path to your data directory
    base_path = r"p:\M48 FYP FEA\Compression Data Analysis\FYP-Cyclic-Compression-Data-Analysis"
    
    print("Starting cyclic compression data analysis...")
    print("Creating output directories...")
    
    # Create organized output directories
    output_base = create_output_directories()
    print(f"Output directories created in: {output_base}")
    
    print("Loading Week 0 data...")
    
    # Load and analyze week 0 data
    data_by_combination = analyze_week0_data(base_path)
    
    print(f"Found {len(data_by_combination)} Filament-SVF-VS combinations")
    for combo_key, combo_data in data_by_combination.items():
        filament, svf, vs = combo_data['filament'], combo_data['svf'], combo_data['vs']
        filament_name = get_filament_name(filament)
        has_normal = 'normal' in combo_data and combo_data['normal']
        has_reprint = 'reprint' in combo_data and combo_data['reprint']
        
        # Show calculated VS values
        normal_vs = combo_data.get('normal', {}).get('calculated_vs', 'N/A')
        reprint_vs = combo_data.get('reprint', {}).get('calculated_vs', 'N/A')
        
        print(f"  {combo_key}: {filament_name}, SVF {svf}%, VS={vs} N/mm (calculated), Normal={has_normal} (VS={normal_vs}), Reprint={has_reprint} (VS={reprint_vs})")
    
    # Group similar vertical stiffnesses (within 40 N/mm)
    print("\nGrouping similar vertical stiffnesses (within 40 N/mm)...")
    vs_groups = group_similar_vs(data_by_combination, tolerance=40)
    print(f"Found {len(vs_groups)} VS groups:")
    for i, group in enumerate(vs_groups):
        vs_range = f"{min(group)}-{max(group)}" if len(group) > 1 else str(group[0])
        print(f"  Group {i+1}: {vs_range} N/mm (values: {sorted(group)})")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    print("Plot 1: VS groups with different combinations (similar stiffnesses within 40 N/mm)...")
    plot_vs_groups(data_by_combination, vs_groups)
    
    print("Plot 2: Combined average by vertical stiffness (all tests averaged)...")
    plot_combined_average_by_vs(data_by_combination)
    
    print("Plot 3: Force vs displacement for specific cycles (1st, 100th, 1000th)...")
    plot_specific_cycles_force_displacement(data_by_combination, [1, 100, 1000])
    
    print("Plot 4: Peak force vs cycle grouped by filament type (same SVF, different TPU types)...")
    plot_by_filament_type(data_by_combination)
    
    print("Plot 5: Peak force vs cycle grouped by SVF percentage (same TPU filament, different SVF)...")
    plot_by_svf_percentage(data_by_combination)
    
    print("Plot 6: Specific cycles (1st, 100th, 1000th) force comparison by VS groups...")
    plot_specific_cycles_by_vs_groups(data_by_combination, vs_groups, [1, 100, 1000])
    
    print(f"\nAnalysis complete! All plots saved in: {output_base}")
    print("\nFolder structure:")
    print("  - Test 1 vertical_stiffness_comparison/: VS groups with different combinations (within 40 N/mm)")
    print("  - Test 1 force_displacement_cycles/: Force-displacement cycle plots")
    print("  - Test 1 force_displacement_by_vs_groups/: Separate VS group folders with individual combination curves")
    print("  - Test 1 by_filament_type/: TPU filament comparison plots (same SVF, different filaments)")
    print("  - Test 1 by_svf_percentage/: SVF comparison plots (same TPU filament, different SVFs)")

if __name__ == "__main__":
    main()