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

def create_output_directories(test_name):
    """
    Create output directories for organizing plots
    """
    base_dir = Path(f'{test_name} analysis_results')
    dirs = [
        base_dir / f'{test_name} vertical_stiffness_comparison',
        base_dir / f'{test_name} force_displacement_cycles',
        base_dir / f'{test_name} force_displacement_by_vs_groups',
        base_dir / f'{test_name} by_filament_type',
        base_dir / f'{test_name} by_svf_percentage'
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
    - '20250901 95 20' -> Filament=95, SVF=20
    - '20250916 87-20 Reprint' -> Filament=87, SVF=20
    """
    parts = folder_name.split()
    if len(parts) >= 2:
        # Handle hyphenated format (e.g., '95-35', '87-20')
        filament_svf = parts[1]
        if '-' in filament_svf:
            filament, svf = filament_svf.split('-')
            # Fix typo: convert 21 to 20 for consistency
            if svf == '21':
                svf = '20'
            return int(filament), int(svf)
        # Handle space-separated format (e.g., '95 20', '87 35')
        elif len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            filament = int(parts[1])
            svf = int(parts[2])
            # Fix typo: convert 21 to 20 for consistency
            if svf == 21:
                svf = 20
            return filament, svf
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

def analyze_week_data(base_path, week_folder_name):
    """
    Analyze week data including normal and reprint folders
    Calculate vertical stiffness from first cycle data
    """
    week_path = Path(base_path) / week_folder_name
    
    # Dictionary to store data organized by Filament-SVF-VS combination
    data_by_combination = {}
    
    # Find all folders in the specified week
    for folder in week_path.iterdir():
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

def plot_specific_cycles_force_displacement(data_by_combination, test_name, cycles_of_interest=[1, 100, 1000]):
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
        ax.set_title(f'{test_name} Force vs Displacement (Averaged over two sets) - {filament_name}, SVF {svf}%\n(Cycles 1, 100, 1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show compression (negative values)
        ax.set_xlim(-7, 1)  # Show from -7mm to +1mm for better visualization
        
        plt.tight_layout()
        
        # Save individual combination plot with Filament and SVF identification
        filament_name_file = get_filament_name(filament)
        output_path = Path(f'{test_name} analysis_results/{test_name} force_displacement_cycles/{test_name} {filament_name_file}_SVF_{svf}_percent_force_displacement_cycles.png')
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

def plot_by_filament_type(data_by_combination, test_name):
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
        ax.set_title(f'{test_name} Peak Force vs Cycle (Averaged across two sets) - SVF {svf}%\n(Comparison across Filament Types)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add SVF annotation
        ax.text(0.02, 0.98, f'SVF {svf}%', transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                verticalalignment='top', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(f'{test_name} analysis_results/{test_name} by_filament_type/{test_name} SVF_{svf}_percent_filament_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Filament comparison plot for SVF {svf}% saved as: {output_path}")

def plot_by_svf_percentage(data_by_combination, test_name):
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
        ax.set_title(f'{test_name} Peak Force vs Cycle (Averaged over two sets) - {filament_name}\n(Comparison across SVF Percentages)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add filament annotation
        ax.text(0.02, 0.98, f'{filament_name}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9),
                verticalalignment='top', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filament_name_file = get_filament_name(filament)
        output_path = Path(f'{test_name} analysis_results/{test_name} by_svf_percentage/{test_name} {filament_name_file}_SVF_comparison.png')
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

def plot_cross_test_peak_force_comparison(base_path):
    """
    Plot peak force vs cycle for every filament-SVF combination across all tests
    Creates one plot per filament-SVF combination showing all test results
    """
    print("\nGenerating cross-test peak force comparison plots...")
    
    # Define all test weeks
    test_weeks = ['0 WK', '1 WK Repeats', '2 WK Repeats', '3 WK Repeats']
    
    # Collect data from all tests
    all_test_data = {}
    
    for week_folder in test_weeks:
        test_name = get_test_name(week_folder)
        week_path = Path(base_path) / week_folder
        
        if week_path.exists():
            print(f"  Loading data from {test_name} ({week_folder})...")
            data_by_combination = analyze_week_data(base_path, week_folder)
            
            if data_by_combination:
                all_test_data[test_name] = data_by_combination
    
    if not all_test_data:
        print("  No data found across all tests, skipping cross-test comparison...")
        return
    
    # Create output directory
    output_dir = Path('Cross Test Comparison analysis_results')
    peak_force_dir = output_dir / 'Cross Test peak_force_vs_cycle'
    peak_force_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all unique filament-SVF combinations across all tests
    all_combinations = set()
    for test_data in all_test_data.values():
        for combo_key, combo_data in test_data.items():
            filament, svf = combo_data['filament'], combo_data['svf']
            all_combinations.add((filament, svf))
    
    print(f"  Found {len(all_combinations)} unique filament-SVF combinations across all tests")
    
    # Create a plot for each filament-SVF combination
    for filament, svf in sorted(all_combinations):
        filament_name = get_filament_name(filament)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Colors for different tests
        test_colors = {'Test 1': 'blue', 'Test 2': 'red', 'Test 3': 'green', 'Test 4': 'orange'}
        
        plot_created = False
        
        # Plot data from each test
        for test_name, test_data in all_test_data.items():
            # Find matching combination in this test
            matching_combo = None
            for combo_key, combo_data in test_data.items():
                if combo_data['filament'] == filament and combo_data['svf'] == svf:
                    matching_combo = combo_data
                    break
            
            if matching_combo:
                color = test_colors.get(test_name, 'black')
                
                # Collect all test data (normal and reprint) for this combination
                all_cycles = []
                all_peak_forces = []
                
                # Add normal test data
                if 'normal' in matching_combo and matching_combo['normal']:
                    cycles = matching_combo['normal']['cycles']
                    peak_forces = matching_combo['normal']['peak_forces']
                    all_cycles.extend(cycles)
                    all_peak_forces.extend(peak_forces)
                
                # Add reprint test data
                if 'reprint' in matching_combo and matching_combo['reprint']:
                    cycles = matching_combo['reprint']['cycles']
                    peak_forces = matching_combo['reprint']['peak_forces']
                    all_cycles.extend(cycles)
                    all_peak_forces.extend(peak_forces)
                
                if all_cycles:
                    # Use exact same binning approach as individual test plots
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
                               '-', color=color, label=f'{test_name}', 
                               linewidth=3, alpha=0.8)
                        plot_created = True
        
        if plot_created:
            ax.set_xlabel('Cycle Number', fontsize=12)
            ax.set_ylabel('Average Peak Force (kN)', fontsize=12)
            ax.set_title(f'All Tests Peak Force vs Cycle (Averaged across two sets)\n{filament_name}, SVF {svf}%', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add combination annotation
            ax.text(0.02, 0.98, f'{filament_name}\nSVF {svf}%', transform=ax.transAxes, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
                    verticalalignment='top', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            output_path = peak_force_dir / f'Cross_Test_{filament_name}_SVF_{svf}_percent_peak_force_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Cross-test peak force plot for {filament_name}, SVF {svf}% saved as: {output_path}")
        else:
            plt.close()
            print(f"    No data found for {filament_name}, SVF {svf}% across tests")
    
    print(f"  Cross-test peak force comparison plots saved in: {peak_force_dir}")

def plot_cross_test_cycle_comparison(base_path, target_cycles=[1, 100, 1000]):
    """
    Plot specific cycles (1st, 100th, 1000th) for every filament-SVF combination across all tests
    Creates separate plots for each cycle AND each filament-SVF combination
    Each combination gets its own plot for each cycle showing all test results
    """
    print("\nGenerating cross-test cycle comparison plots...")
    
    # Define all test weeks
    test_weeks = ['0 WK', '1 WK Repeats', '2 WK Repeats', '3 WK Repeats']
    
    # Collect data from all tests
    all_test_data = {}
    
    for week_folder in test_weeks:
        test_name = get_test_name(week_folder)
        week_path = Path(base_path) / week_folder
        
        if week_path.exists():
            print(f"  Loading data from {test_name} ({week_folder})...")
            data_by_combination = analyze_week_data(base_path, week_folder)
            
            if data_by_combination:
                all_test_data[test_name] = data_by_combination
    
    if not all_test_data:
        print("  No data found across all tests, skipping cross-test cycle comparison...")
        return
    
    # Create output directory
    output_dir = Path('Cross Test Comparison analysis_results')
    cycle_comparison_dir = output_dir / 'Cross Test cycle_comparisons'
    cycle_comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all unique filament-SVF combinations across all tests
    all_combinations = set()
    for test_data in all_test_data.values():
        for combo_key, combo_data in test_data.items():
            filament, svf = combo_data['filament'], combo_data['svf']
            all_combinations.add((filament, svf))
    
    print(f"  Found {len(all_combinations)} unique filament-SVF combinations across all tests")
    
    # Helper function for averaging hysteresis loops
    def average_hysteresis_loops(displacements_list, forces_list):
        if not displacements_list or not forces_list:
            return None, None
        
        processed_data = []
        for disp, force in zip(displacements_list, forces_list):
            valid_mask = ~(np.isnan(disp) | np.isnan(force))
            disp_clean = disp[valid_mask]
            force_clean = force[valid_mask]
            
            if len(disp_clean) < 10:
                continue
            
            min_disp = np.min(disp_clean)
            max_disp = np.max(disp_clean)
            disp_normalized = disp_clean - max_disp
            
            disp_range = abs(min_disp - max_disp)
            if disp_range > 0.1:
                target_range = 6.0
                scale_factor = target_range / disp_range
                disp_normalized = disp_normalized * scale_factor
            
            processed_data.append((disp_normalized, force_clean))
        
        if len(processed_data) < 1:
            return None, None
        
        if len(processed_data) == 1:
            return processed_data[0]
        
        ref_idx = np.argmax([len(data[0]) for data in processed_data])
        ref_disp, ref_force = processed_data[ref_idx]
        
        all_forces = [ref_force]
        
        for i, (disp, force) in enumerate(processed_data):
            if i == ref_idx:
                continue
                
            ref_time = np.linspace(0, 1, len(ref_force))
            data_time = np.linspace(0, 1, len(force))
            interp_force = np.interp(ref_time, data_time, force)
            all_forces.append(interp_force)
        
        avg_force = np.mean(all_forces, axis=0)
        return ref_disp, avg_force
    
    # Create plots for each target cycle and each combination
    for target_cycle in target_cycles:
        print(f"    Processing cycle {target_cycle}...")
        
        # Create a subdirectory for this cycle
        cycle_dir = cycle_comparison_dir / f'Cycle_{target_cycle}'
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a separate plot for each filament-SVF combination
        for filament, svf in sorted(all_combinations):
            filament_name = get_filament_name(filament)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Colors for different tests
            test_colors = {'Test 1': 'blue', 'Test 2': 'red', 'Test 3': 'green', 'Test 4': 'orange'}
            
            plot_created = False
            
            # Plot data from each test for this specific combination
            for test_name, test_data in all_test_data.items():
                # Find matching combination in this test
                matching_combo = None
                for combo_key, combo_data in test_data.items():
                    if combo_data['filament'] == filament and combo_data['svf'] == svf:
                        matching_combo = combo_data
                        break
                
                if matching_combo:
                    # Collect force-displacement data for target cycle
                    all_displacements = []
                    all_forces = []
                    
                    # Get normal test data
                    if 'normal' in matching_combo and matching_combo['normal'] and 'full_data' in matching_combo['normal']:
                        df_normal = matching_combo['normal']['full_data']
                        cycle_data = df_normal[df_normal['Total Cycles'] == target_cycle]
                        if not cycle_data.empty:
                            displacement = cycle_data['Displacement(Linear:Digital Position) (mm)'].values
                            force = cycle_data['Force(Linear:Load) (kN)'].values
                            if len(displacement) > 5 and len(force) > 5:
                                all_displacements.append(displacement)
                                all_forces.append(force)
                    
                    # Get reprint test data
                    if 'reprint' in matching_combo and matching_combo['reprint'] and 'full_data' in matching_combo['reprint']:
                        df_reprint = matching_combo['reprint']['full_data']
                        cycle_data = df_reprint[df_reprint['Total Cycles'] == target_cycle]
                        if not cycle_data.empty:
                            displacement = cycle_data['Displacement(Linear:Digital Position) (mm)'].values
                            force = cycle_data['Force(Linear:Load) (kN)'].values
                            if len(displacement) > 5 and len(force) > 5:
                                all_displacements.append(displacement)
                                all_forces.append(force)
                    
                    # Calculate and plot average force-displacement curve
                    if all_displacements and all_forces:
                        avg_disp, avg_force = average_hysteresis_loops(all_displacements, all_forces)
                        
                        if avg_disp is not None and avg_force is not None:
                            color = test_colors.get(test_name, 'black')
                            
                            ax.plot(avg_disp, avg_force, 
                                   color=color, linewidth=3, alpha=0.8, 
                                   label=f'{test_name}', marker='o', markersize=2)
                            
                            plot_created = True
            
            if plot_created:
                ax.set_xlabel('Displacement (mm)', fontsize=12)
                ax.set_ylabel('Force (kN)', fontsize=12)
                ax.set_title(f'All Tests Force vs Displacement (Averaged across two sets) - Cycle {target_cycle}\n{filament_name}, SVF {svf}%', 
                            fontsize=14)
                
                ax.legend(fontsize=11, loc='best')
                ax.grid(True, alpha=0.3)
                
                # Add combination annotation
                ax.text(0.02, 0.98, f'{filament_name}\nSVF {svf}%\nCycle {target_cycle}', 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
                       verticalalignment='top', fontsize=10, fontweight='bold')
                
                # Set x-axis to show compression
                ax.set_xlim(-7, 1)
                
                plt.tight_layout()
                
                # Save plot for this specific combination and cycle
                output_path = cycle_dir / f'Cross_Test_Cycle_{target_cycle}_{filament_name}_SVF_{svf}_percent_comparison.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"      Cross-test cycle {target_cycle} plot for {filament_name}, SVF {svf}% saved as: {output_path}")
            else:
                plt.close()
                print(f"      No data found for {filament_name}, SVF {svf}% in cycle {target_cycle}")
    
    print(f"  Cross-test cycle comparison plots saved in: {cycle_comparison_dir}")
    print(f"  Organization: Each cycle has its own subdirectory with separate plots for each filament-SVF combination")

def plot_cross_test_vs_comparison(base_path):
    """
    Compare peak force vs cycle curves (averaged over two sets) between similar vertical stiffnesses 
    (within 40 N/mm) across all tests. Creates plots showing how different VS groups perform 
    across all test conditions.
    """
    print("\nGenerating cross-test vertical stiffness comparison plots...")
    
    # Define all test weeks
    test_weeks = ['0 WK', '1 WK Repeats', '2 WK Repeats', '3 WK Repeats']
    
    # Collect data from all tests with vertical stiffness information
    all_test_data = {}
    all_vs_values = set()
    
    for week_folder in test_weeks:
        test_name = get_test_name(week_folder)
        week_path = Path(base_path) / week_folder
        
        if week_path.exists():
            print(f"  Loading data from {test_name} ({week_folder})...")
            data_by_combination = analyze_week_data(base_path, week_folder)
            
            if data_by_combination:
                all_test_data[test_name] = data_by_combination
                # Collect all VS values
                for combo_data in data_by_combination.values():
                    all_vs_values.add(combo_data['vs'])
    
    if not all_test_data:
        print("  No data found across all tests, skipping cross-test VS comparison...")
        return
    
    # Group similar vertical stiffnesses across all tests (within 40 N/mm tolerance)
    all_vs_list = sorted(list(all_vs_values))
    vs_groups = []
    used_vs = set()
    
    for vs in all_vs_list:
        if vs in used_vs:
            continue
            
        # Find all VS values within 40 N/mm tolerance
        group = [vs]
        used_vs.add(vs)
        
        for other_vs in all_vs_list:
            if other_vs != vs and other_vs not in used_vs:
                if abs(vs - other_vs) <= 40:  # 40 N/mm tolerance
                    group.append(other_vs)
                    used_vs.add(other_vs)
        
        vs_groups.append(sorted(group))
    
    print(f"  Found {len(vs_groups)} vertical stiffness groups across all tests:")
    for i, group in enumerate(vs_groups):
        vs_range = f"{min(group)}-{max(group)}" if len(group) > 1 else str(group[0])
        print(f"    VS Group {i+1}: {vs_range} N/mm ({len(group)} values)")
    
    # Create output directory
    output_dir = Path('Cross Test Comparison analysis_results')
    vs_comparison_dir = output_dir / 'Cross Test vertical_stiffness_comparison'
    vs_comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each VS group
    for group_idx, vs_group in enumerate(vs_groups):
        # Check if this group has different combinations (different filament OR SVF)
        # Collect all combinations in this VS group across all tests
        combinations_in_group = []
        test_combination_count = {}  # Track which tests have which combinations
        
        for test_name, test_data in all_test_data.items():
            test_combination_count[test_name] = []
            for combo_key, combo_data in test_data.items():
                if combo_data['vs'] in vs_group:
                    filament, svf = combo_data['filament'], combo_data['svf']
                    combinations_in_group.append((filament, svf))
                    test_combination_count[test_name].append((filament, svf))
        
        # Check if we have different combinations (not just same filament-SVF with different VS)
        unique_combinations = set(combinations_in_group)
        if len(unique_combinations) <= 1:
            print(f"  Skipping VS Group {group_idx+1} - only contains identical filament-SVF combinations")
            continue
        
        # Check that we have at least 2 tests with data for this VS group
        tests_with_data = sum(1 for test_combos in test_combination_count.values() if len(test_combos) > 0)
        if tests_with_data < 2:
            print(f"  Skipping VS Group {group_idx+1} - appears in fewer than 2 tests ({tests_with_data} test(s))")
            continue
        
        # Check that we have sufficient combinations across tests
        total_data_points = len(combinations_in_group)
        if total_data_points < 4:  # Need at least 4 data points for meaningful comparison
            print(f"  Skipping VS Group {group_idx+1} - insufficient data points ({total_data_points} total)")
            continue
        
        print(f"  Processing VS Group {group_idx+1} with {len(unique_combinations)} unique combinations across {tests_with_data} tests")
        for test_name, combos in test_combination_count.items():
            if combos:
                combo_names = [f"{get_filament_name(f)}-SVF{s}%" for f, s in set(combos)]
                print(f"    {test_name}: {', '.join(combo_names)}")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Colors for different tests
        test_colors = {'Test 1': 'blue', 'Test 2': 'red', 'Test 3': 'green', 'Test 4': 'orange'}
        
        plot_created = False
        
        # Plot data for each test and each individual combination
        line_styles = ['-', '--', '-.', ':']
        
        for test_name, test_data in all_test_data.items():
            test_color = test_colors.get(test_name, 'black')
            
            # Plot each combination individually for this test
            combo_idx = 0
            for combo_key, combo_data in test_data.items():
                if combo_data['vs'] in vs_group:
                    filament, svf = combo_data['filament'], combo_data['svf']
                    filament_name = get_filament_name(filament)
                    combo_label = f"{filament_name}-SVF{svf}%"
                    
                    # Combine normal and reprint data for this specific combination
                    all_cycles_for_combo = []
                    all_forces_for_combo = []
                    
                    # Add normal test data
                    if 'normal' in combo_data and combo_data['normal']:
                        cycles = combo_data['normal']['cycles']
                        forces = combo_data['normal']['peak_forces']
                        all_cycles_for_combo.extend(cycles)
                        all_forces_for_combo.extend(forces)
                    
                    # Add reprint test data  
                    if 'reprint' in combo_data and combo_data['reprint']:
                        cycles = combo_data['reprint']['cycles']
                        forces = combo_data['reprint']['peak_forces']
                        all_cycles_for_combo.extend(cycles)
                        all_forces_for_combo.extend(forces)
                    
                    if all_cycles_for_combo and all_forces_for_combo:
                        # Use same binning approach as other functions
                        max_cycle = max(all_cycles_for_combo)
                        bin_edges = np.linspace(1, max_cycle, min(100, max_cycle))
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        binned_forces = []
                        for i in range(len(bin_edges) - 1):
                            mask = (np.array(all_cycles_for_combo) >= bin_edges[i]) & (np.array(all_cycles_for_combo) < bin_edges[i+1])
                            if np.any(mask):
                                binned_forces.append(np.mean(np.array(all_forces_for_combo)[mask]))
                            else:
                                binned_forces.append(np.nan)
                        
                        # Remove NaN values for plotting
                        valid_mask = ~np.isnan(binned_forces)
                        if np.any(valid_mask):
                            # Use different line styles for different combinations
                            line_style = line_styles[combo_idx % len(line_styles)]
                            
                            # Create label for this specific combination
                            normal_count = len(combo_data.get('normal', {}).get('cycles', [])) if 'normal' in combo_data else 0
                            reprint_count = len(combo_data.get('reprint', {}).get('cycles', [])) if 'reprint' in combo_data else 0
                            set_info = f"N+R" if normal_count > 0 and reprint_count > 0 else ("N" if normal_count > 0 else "R")
                            
                            label = f'{test_name} - {combo_label} (VS={combo_data["vs"]:.1f}, {set_info})'
                            
                            ax.plot(bin_centers[valid_mask], np.array(binned_forces)[valid_mask], 
                                   line_style, color=test_color, label=label, 
                                   linewidth=2, alpha=0.8, marker='o', markersize=3)
                            plot_created = True
                            combo_idx += 1
        
        if plot_created:
            ax.set_xlabel('Cycle Number', fontsize=12)
            ax.set_ylabel('Average Peak Force (kN)', fontsize=12)
            
            # Create title with VS range and combination info
            vs_range = f"{min(vs_group)}-{max(vs_group)}" if len(vs_group) > 1 else str(vs_group[0])
            combo_summary = ", ".join([f"{get_filament_name(f)}-SVF {s}%" for f, s in sorted(unique_combinations)])
            ax.set_title(f'Cross-Test Peak Force vs Cycle Comparison (Individual Combinations, Averaged over two sets)\nSimilar Vertical Stiffnesses: {vs_range} N/mm\nCombinations: {combo_summary}', 
                        fontsize=14)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add VS group annotation
            ax.text(0.02, 0.98, f'VS Group {group_idx+1}\n{vs_range} N/mm\n(Individual Lines)', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9),
                   verticalalignment='top', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            output_path = vs_comparison_dir / f'Cross_Test_VS_Group_{group_idx+1}_{vs_range}_Nmm_individual_combinations.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Cross-test VS comparison for Group {group_idx+1} ({vs_range} N/mm) saved as: {output_path}")
        else:
            plt.close()
            print(f"  No data found for VS Group {group_idx+1} ({vs_range} N/mm)")
    
    print(f"  Cross-test vertical stiffness comparison plots saved in: {vs_comparison_dir}")

def get_test_name(folder_name):
    """
    Convert folder name to test name
    """
    test_name_map = {
        '0 WK': 'Test 1',
        '1 WK Repeats': 'Test 2',
        '2 WK Repeats': 'Test 3', 
        '3 WK Repeats': 'Test 4'
    }
    return test_name_map.get(folder_name, f'Test {folder_name}')

def plot_first_cycle_peak_force_histogram(csv_path):
    """
    Create a histogram figure showing each test's peak force for cycle 1 from the first week first cycle CSV,
    with relative standard deviation overlayed for each combination set (normal and reprint)
    """
    print("\nCreating first cycle peak force histogram...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Calculate peak force (absolute value) for each combination and test type
        peak_data = df.groupby(['Filament_Name', 'SVF_Percentage', 'Test_Type'])['Force(Linear:Load) (kN)'].apply(
            lambda x: np.max(np.abs(x))
        ).reset_index()
        peak_data.columns = ['Filament_Name', 'SVF_Percentage', 'Test_Type', 'Peak_Force_kN']
        
        # Calculate statistics for each combination (normal and reprint together)
        stats_data = []
        for (filament, svf), group in peak_data.groupby(['Filament_Name', 'SVF_Percentage']):
            normal_peak = group[group['Test_Type'] == 'normal']['Peak_Force_kN'].values
            reprint_peak = group[group['Test_Type'] == 'reprint']['Peak_Force_kN'].values
            
            all_peaks = []
            if len(normal_peak) > 0:
                all_peaks.append(normal_peak[0])
            if len(reprint_peak) > 0:
                all_peaks.append(reprint_peak[0])
            
            if len(all_peaks) >= 2:  # Both normal and reprint exist
                mean_peak = np.mean(all_peaks)
                std_peak = np.std(all_peaks, ddof=1)
                rsd = (std_peak / mean_peak) * 100  # Relative standard deviation as percentage
                
                stats_data.append({
                    'Filament_Name': filament,
                    'SVF_Percentage': svf,
                    'Mean_Peak_Force_kN': mean_peak,
                    'Std_Peak_Force_kN': std_peak,
                    'RSD_Percent': rsd,
                    'Normal_Peak': normal_peak[0] if len(normal_peak) > 0 else np.nan,
                    'Reprint_Peak': reprint_peak[0] if len(reprint_peak) > 0 else np.nan,
                    'Count': len(all_peaks)
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        if stats_df.empty:
            print("No data found for histogram creation.")
            return None
        
        # Create the histogram plot
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        
        # Prepare data for plotting
        combinations = []
        mean_forces = []
        rsd_values = []
        normal_forces = []
        reprint_forces = []
        
        for _, row in stats_df.iterrows():
            combo_label = f"{row['Filament_Name']}\nSVF {row['SVF_Percentage']}%"
            combinations.append(combo_label)
            mean_forces.append(row['Mean_Peak_Force_kN'])
            rsd_values.append(row['RSD_Percent'])
            normal_forces.append(row['Normal_Peak'])
            reprint_forces.append(row['Reprint_Peak'])
        
        x_pos = np.arange(len(combinations))
        
        # Create grouped bar chart for individual test values
        width = 0.35
        bars1 = ax1.bar(x_pos - width/2, normal_forces, width, label='Normal Test', 
                       color='lightblue', alpha=0.8, edgecolor='darkblue')
        bars2 = ax1.bar(x_pos + width/2, reprint_forces, width, label='Reprint Test', 
                       color='lightcoral', alpha=0.8, edgecolor='darkred')
        
        # Add mean values as points
        ax1.plot(x_pos, mean_forces, 'ko', markersize=8, label='Mean Peak Force', zorder=3)
        
        # Add error bars representing standard deviation
        ax1.errorbar(x_pos, mean_forces, yerr=[row['Std_Peak_Force_kN'] for _, row in stats_df.iterrows()], 
                    fmt='none', color='black', capsize=5, capthick=2, zorder=3)
        
        # Set up primary y-axis (force)
        ax1.set_xlabel('Filament-SVF Combinations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Peak Force (kN)', fontsize=12, fontweight='bold')
        ax1.set_title('First Week First Cycle Peak Force Analysis\nPeak Force by Combination with Individual RSD Calculations', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(combinations, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper left')
        
        # Create secondary y-axis for RSD
        ax2 = ax1.twinx()
        
        # Plot individual RSD chains for each combination (not connected)
        # Each combination gets its own separate visual indicator
        colors = plt.cm.Set3(np.linspace(0, 1, len(combinations)))
        
        for i, (x, rsd, normal, reprint) in enumerate(zip(x_pos, rsd_values, normal_forces, reprint_forces)):
            # Calculate RSD range for visual representation on secondary axis
            combo_color = colors[i]
            
            # Convert force values to RSD axis scale for visualization
            # We'll show the individual normal and reprint values as points on the RSD axis
            # scaled relative to their contribution to the RSD calculation
            mean_val = (normal + reprint) / 2
            rsd_normal = ((normal - mean_val) / mean_val) * 100 + rsd  # Offset from RSD center
            rsd_reprint = ((reprint - mean_val) / mean_val) * 100 + rsd  # Offset from RSD center
            
            # Plot the RSD chain for this combination (2 points connected)
            ax2.plot([x-0.1, x+0.1], [rsd_normal, rsd_reprint], 
                    color=combo_color, linewidth=2, alpha=0.7, zorder=2)
            
            # Plot individual points
            ax2.plot(x-0.1, rsd_normal, 'o', color=combo_color, markersize=5, 
                    alpha=0.8, zorder=3, label='_nolegend_')
            ax2.plot(x+0.1, rsd_reprint, 's', color=combo_color, markersize=5, 
                    alpha=0.8, zorder=3, label='_nolegend_')
            
            # Add RSD value label
            ax2.annotate(f'{rsd:.1f}%', (x, rsd), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=9, color='red', fontweight='bold')
        
        ax2.set_ylabel('Relative Standard Deviation (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add custom legend for RSD visualization
        from matplotlib.lines import Line2D
        rsd_legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='RSD Chain (Normal ↔ Reprint)'),
            Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=5, alpha=0.8, label='Normal Test'),
            Line2D([0], [0], marker='s', color='gray', linewidth=0, markersize=5, alpha=0.8, label='Reprint Test')
        ]
        ax2.legend(handles=rsd_legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path('First_Week_First_Cycle_Peak_Force_Histogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"First cycle peak force histogram saved as: {output_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Number of combinations analyzed: {len(stats_df)}")
        print(f"Mean peak force range: {stats_df['Mean_Peak_Force_kN'].min():.3f} - {stats_df['Mean_Peak_Force_kN'].max():.3f} kN")
        print(f"RSD range: {stats_df['RSD_Percent'].min():.1f}% - {stats_df['RSD_Percent'].max():.1f}%")
        print(f"Average RSD: {stats_df['RSD_Percent'].mean():.1f}%")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating histogram: {e}")
        return None

def plot_normal_vs_reprint_comparison(data_by_combination, test_name):
    """
    Create plots comparing peak force vs cycle between normal and reprint sets for each combination
    """
    print(f"\nCreating normal vs reprint comparison plots for {test_name}...")
    
    # Create output directory for comparison plots
    comparison_dir = Path(f'{test_name} analysis_results/{test_name} normal_vs_reprint_comparison')
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each filament-SVF combination
    for combination_key, combo_data in data_by_combination.items():
        filament, svf, vs = combo_data['filament'], combo_data['svf'], combo_data['vs']
        filament_name = get_filament_name(filament)
        
        # Check if both normal and reprint data exist
        has_normal = 'normal' in combo_data and combo_data['normal']
        has_reprint = 'reprint' in combo_data and combo_data['reprint']
        
        if not (has_normal and has_reprint):
            print(f"  Skipping {filament_name} SVF {svf}% - missing normal or reprint data")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot normal test data
        if has_normal:
            normal_cycles = combo_data['normal']['cycles']
            normal_forces = combo_data['normal']['peak_forces']
            # Remove any zero or invalid values and ensure proper data range
            valid_normal = [(c, f) for c, f in zip(normal_cycles, normal_forces) if f > 0 and not np.isnan(f)]
            if valid_normal:
                normal_cycles_clean, normal_forces_clean = zip(*valid_normal)
                ax.plot(normal_cycles_clean, normal_forces_clean, 'b-o', linewidth=2, markersize=4, 
                       label='Normal Test', alpha=0.8)
        
        # Plot reprint test data
        if has_reprint:
            reprint_cycles = combo_data['reprint']['cycles']
            reprint_forces = combo_data['reprint']['peak_forces']
            # Remove any zero or invalid values and ensure proper data range
            valid_reprint = [(c, f) for c, f in zip(reprint_cycles, reprint_forces) if f > 0 and not np.isnan(f)]
            if valid_reprint:
                reprint_cycles_clean, reprint_forces_clean = zip(*valid_reprint)
                ax.plot(reprint_cycles_clean, reprint_forces_clean, 'r-s', linewidth=2, markersize=4, 
                       label='Reprint Test', alpha=0.8)
        
        # Calculate and show statistics
        if has_normal and has_reprint:
            # Use cleaned data for statistics
            normal_forces_clean = [f for c, f in zip(normal_cycles, normal_forces) if f > 0 and not np.isnan(f)]
            reprint_forces_clean = [f for c, f in zip(reprint_cycles, reprint_forces) if f > 0 and not np.isnan(f)]
            
            # Find common cycle range for comparison
            max_cycles = min(len(normal_forces_clean), len(reprint_forces_clean))
            if max_cycles > 0:
                # Calculate average difference over the cycle range
                common_cycles = min(max_cycles, 100)  # Compare first 100 cycles or available cycles
                normal_avg = np.mean(normal_forces_clean[:common_cycles])
                reprint_avg = np.mean(reprint_forces_clean[:common_cycles])
                difference = abs(normal_avg - reprint_avg)
                relative_diff = (difference / normal_avg) * 100
                
                # Add statistics text box
                stats_text = f'Average Peak Force (first {common_cycles} cycles):\n'
                stats_text += f'Normal: {normal_avg:.4f} kN\n'
                stats_text += f'Reprint: {reprint_avg:.4f} kN\n'
                stats_text += f'Difference: {difference:.4f} kN ({relative_diff:.2f}%)\n'
                stats_text += f'Vertical Stiffness: {vs} N/mm'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9),
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        # Formatting
        ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Force (kN)', fontsize=12, fontweight='bold')
        ax.set_title(f'{test_name} Normal vs Reprint Peak Force Comparison\n{filament_name}, SVF {svf}%', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add combination annotation
        ax.text(0.98, 0.02, f'{filament_name}\nSVF {svf}%', transform=ax.transAxes, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
               verticalalignment='bottom', horizontalalignment='right', 
               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filament_name_file = get_filament_name(filament)
        output_path = comparison_dir / f'{test_name}_{filament_name_file}_SVF_{svf}_percent_normal_vs_reprint.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Normal vs reprint comparison for {filament_name} SVF {svf}% saved as: {output_path}")
    
    print(f"Normal vs reprint comparison plots saved in: {comparison_dir}")
    return comparison_dir

def plot_first_cycle_vertical_stiffness_histogram(csv_path):
    """
    Create a histogram figure showing each test's vertical stiffness from the first week first cycle CSV,
    with relative standard deviation overlayed for each combination set (normal and reprint)
    """
    print("\nCreating first cycle vertical stiffness histogram...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Get vertical stiffness data for each combination and test type
        vs_data = df.groupby(['Filament_Name', 'SVF_Percentage', 'Test_Type'])['Calculated_Vertical_Stiffness_N_per_mm'].first().reset_index()
        vs_data.columns = ['Filament_Name', 'SVF_Percentage', 'Test_Type', 'Vertical_Stiffness_N_per_mm']
        
        # Remove any NaN or invalid stiffness values
        vs_data = vs_data.dropna(subset=['Vertical_Stiffness_N_per_mm'])
        vs_data = vs_data[vs_data['Vertical_Stiffness_N_per_mm'] > 0]  # Only positive stiffness values
        
        if vs_data.empty:
            print("No valid vertical stiffness data found for histogram creation.")
            return None
        
        # Calculate statistics for each combination (normal and reprint together)
        stats_data = []
        for (filament, svf), group in vs_data.groupby(['Filament_Name', 'SVF_Percentage']):
            normal_vs = group[group['Test_Type'] == 'normal']['Vertical_Stiffness_N_per_mm'].values
            reprint_vs = group[group['Test_Type'] == 'reprint']['Vertical_Stiffness_N_per_mm'].values
            
            all_vs = []
            if len(normal_vs) > 0:
                all_vs.append(normal_vs[0])
            if len(reprint_vs) > 0:
                all_vs.append(reprint_vs[0])
            
            if len(all_vs) >= 2:  # Both normal and reprint exist
                mean_vs = np.mean(all_vs)
                std_vs = np.std(all_vs, ddof=1)
                rsd = (std_vs / mean_vs) * 100  # Relative standard deviation as percentage
                
                stats_data.append({
                    'Filament_Name': filament,
                    'SVF_Percentage': svf,
                    'Mean_Vertical_Stiffness_N_per_mm': mean_vs,
                    'Std_Vertical_Stiffness_N_per_mm': std_vs,
                    'RSD_Percent': rsd,
                    'Normal_VS': normal_vs[0] if len(normal_vs) > 0 else np.nan,
                    'Reprint_VS': reprint_vs[0] if len(reprint_vs) > 0 else np.nan,
                    'Count': len(all_vs)
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        if stats_df.empty:
            print("No valid combination data found for vertical stiffness histogram creation.")
            return None
        
        # Create the histogram plot
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        
        # Prepare data for plotting
        combinations = []
        mean_vs_values = []
        rsd_values = []
        normal_vs_values = []
        reprint_vs_values = []
        
        for _, row in stats_df.iterrows():
            combo_label = f"{row['Filament_Name']}\nSVF {row['SVF_Percentage']}%"
            combinations.append(combo_label)
            mean_vs_values.append(row['Mean_Vertical_Stiffness_N_per_mm'])
            rsd_values.append(row['RSD_Percent'])
            normal_vs_values.append(row['Normal_VS'])
            reprint_vs_values.append(row['Reprint_VS'])
        
        x_pos = np.arange(len(combinations))
        
        # Create grouped bar chart for individual test values
        width = 0.35
        bars1 = ax1.bar(x_pos - width/2, normal_vs_values, width, label='Normal Test', 
                       color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        bars2 = ax1.bar(x_pos + width/2, reprint_vs_values, width, label='Reprint Test', 
                       color='lightsalmon', alpha=0.8, edgecolor='darkorange')
        
        # Add mean values as points
        ax1.plot(x_pos, mean_vs_values, 'ko', markersize=8, label='Mean Vertical Stiffness', zorder=3)
        
        # Add error bars representing standard deviation
        ax1.errorbar(x_pos, mean_vs_values, yerr=[row['Std_Vertical_Stiffness_N_per_mm'] for _, row in stats_df.iterrows()], 
                    fmt='none', color='black', capsize=5, capthick=2, zorder=3)
        
        # Set up primary y-axis (vertical stiffness)
        ax1.set_xlabel('Filament-SVF Combinations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Vertical Stiffness (N/mm)', fontsize=12, fontweight='bold')
        ax1.set_title('First Week First Cycle Vertical Stiffness Analysis\nVertical Stiffness by Combination with Individual RSD Calculations', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(combinations, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper left')
        
        # Create secondary y-axis for RSD
        ax2 = ax1.twinx()
        
        # Plot individual RSD chains for each combination (not connected)
        # Each combination gets its own separate visual indicator
        colors = plt.cm.Set3(np.linspace(0, 1, len(combinations)))
        
        for i, (x, rsd, normal, reprint) in enumerate(zip(x_pos, rsd_values, normal_vs_values, reprint_vs_values)):
            # Calculate RSD range for visual representation on secondary axis
            combo_color = colors[i]
            
            # Convert VS values to RSD axis scale for visualization
            # We'll show the individual normal and reprint values as points on the RSD axis
            # scaled relative to their contribution to the RSD calculation
            mean_val = (normal + reprint) / 2
            rsd_normal = ((normal - mean_val) / mean_val) * 100 + rsd  # Offset from RSD center
            rsd_reprint = ((reprint - mean_val) / mean_val) * 100 + rsd  # Offset from RSD center
            
            # Plot the RSD chain for this combination (2 points connected)
            ax2.plot([x-0.1, x+0.1], [rsd_normal, rsd_reprint], 
                    color=combo_color, linewidth=2, alpha=0.7, zorder=2)
            
            # Plot individual points
            ax2.plot(x-0.1, rsd_normal, 'o', color=combo_color, markersize=5, 
                    alpha=0.8, zorder=3, label='_nolegend_')
            ax2.plot(x+0.1, rsd_reprint, 's', color=combo_color, markersize=5, 
                    alpha=0.8, zorder=3, label='_nolegend_')
            
            # Add RSD value label
            ax2.annotate(f'{rsd:.1f}%', (x, rsd), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=9, color='red', fontweight='bold')
        
        ax2.set_ylabel('Relative Standard Deviation (%)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add custom legend for RSD visualization
        from matplotlib.lines import Line2D
        rsd_legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='RSD Chain (Normal ↔ Reprint)'),
            Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=5, alpha=0.8, label='Normal Test'),
            Line2D([0], [0], marker='s', color='gray', linewidth=0, markersize=5, alpha=0.8, label='Reprint Test')
        ]
        ax2.legend(handles=rsd_legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path('First_Week_First_Cycle_Vertical_Stiffness_Histogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"First cycle vertical stiffness histogram saved as: {output_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Number of combinations analyzed: {len(stats_df)}")
        print(f"Mean vertical stiffness range: {stats_df['Mean_Vertical_Stiffness_N_per_mm'].min():.1f} - {stats_df['Mean_Vertical_Stiffness_N_per_mm'].max():.1f} N/mm")
        print(f"RSD range: {stats_df['RSD_Percent'].min():.1f}% - {stats_df['RSD_Percent'].max():.1f}%")
        print(f"Average RSD: {stats_df['RSD_Percent'].mean():.1f}%")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating vertical stiffness histogram: {e}")
        return None

def create_first_cycle_combined_csv(base_path):
    """
    Extract first cycle data from all tracking.csv files and create a combined CSV file
    """
    print("\nCreating combined first cycle CSV file...")
    
    # Define all test weeks
    test_weeks = ['0 WK', '1 WK Repeats', '2 WK Repeats', '3 WK Repeats']
    
    # List to store all first cycle data
    all_first_cycle_data = []
    
    # Process each test week
    for week_folder in test_weeks:
        test_name = get_test_name(week_folder)
        week_path = Path(base_path) / week_folder
        
        if not week_path.exists():
            print(f"  Warning: {week_folder} folder not found, skipping...")
            continue
            
        print(f"  Processing {test_name} ({week_folder})...")
        
        # Find all folders in the specified week
        for folder in week_path.iterdir():
            if folder.is_dir():
                filament, svf = extract_parameters_from_filename(folder.name)
                if filament is not None and svf is not None:
                    # Determine if this is a reprint or normal test
                    test_type = "reprint" if "REPRINT" in folder.name.upper() else "normal"
                    
                    # Look for tracking CSV file
                    tracking_file = folder / "Test1" / "Test1.steps.tracking.csv"
                    if tracking_file.exists():
                        try:
                            # Load the CSV file
                            df = pd.read_csv(tracking_file)
                            df.columns = df.columns.str.strip()  # Clean column names
                            
                            # Filter for first cycle only
                            first_cycle_data = df[df['Total Cycles'] == 1].copy()
                            
                            if not first_cycle_data.empty:
                                # Add identifying columns
                                first_cycle_data['Test_Name'] = test_name
                                first_cycle_data['Week_Folder'] = week_folder
                                first_cycle_data['Folder_Name'] = folder.name
                                first_cycle_data['Filament_Type'] = filament
                                first_cycle_data['SVF_Percentage'] = svf
                                first_cycle_data['Test_Type'] = test_type
                                first_cycle_data['Filament_Name'] = get_filament_name(filament)
                                
                                # Calculate vertical stiffness for this first cycle
                                vs_calculated = calculate_vertical_stiffness(first_cycle_data)
                                first_cycle_data['Calculated_Vertical_Stiffness_N_per_mm'] = vs_calculated
                                
                                # Add to the combined list
                                all_first_cycle_data.append(first_cycle_data)
                                
                                print(f"    Added first cycle data from {folder.name} ({len(first_cycle_data)} rows)")
                            else:
                                print(f"    Warning: No first cycle data found in {folder.name}")
                                
                        except Exception as e:
                            print(f"    Error processing {tracking_file}: {e}")
                    else:
                        print(f"    Warning: Tracking file not found in {folder.name}")
    
    # Combine all first cycle data
    if all_first_cycle_data:
        combined_df = pd.concat(all_first_cycle_data, ignore_index=True)
        
        # Reorder columns to put identifying information first
        id_columns = ['Test_Name', 'Week_Folder', 'Folder_Name', 'Filament_Name', 'Filament_Type', 
                     'SVF_Percentage', 'Test_Type', 'Calculated_Vertical_Stiffness_N_per_mm']
        
        # Get original data columns
        original_columns = [col for col in combined_df.columns if col not in id_columns]
        
        # Reorder columns
        final_columns = id_columns + original_columns
        combined_df = combined_df[final_columns]
        
        # Save to CSV
        output_path = Path(base_path) / 'All_Tests_First_Cycle_Combined_Data.csv'
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n  Combined first cycle CSV created successfully!")
        print(f"  Output file: {output_path}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total columns: {len(combined_df.columns)}")
        
        # Print summary statistics
        print(f"\n  Summary by test:")
        test_summary = combined_df.groupby(['Test_Name', 'Test_Type']).size().reset_index(name='Row_Count')
        for _, row in test_summary.iterrows():
            print(f"    {row['Test_Name']} ({row['Test_Type']}): {row['Row_Count']} rows")
            
        print(f"\n  Summary by filament-SVF combination:")
        combo_summary = combined_df.groupby(['Filament_Name', 'SVF_Percentage']).agg({
            'Test_Name': 'nunique',
            'Folder_Name': 'nunique',
            'Calculated_Vertical_Stiffness_N_per_mm': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        combo_summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in combo_summary.columns]
        combo_summary = combo_summary.rename(columns={
            'Test_Name_nunique': 'Tests_Count',
            'Folder_Name_nunique': 'Samples_Count',
            'Calculated_Vertical_Stiffness_N_per_mm_mean': 'Avg_VS_N_per_mm',
            'Calculated_Vertical_Stiffness_N_per_mm_std': 'Std_VS_N_per_mm'
        })
        
        print(combo_summary.to_string())
        
        return output_path
    else:
        print("  No first cycle data found across all tests!")
        return None

def main():
    """
    Main analysis function
    """
    # Set the base path to the directory where this script is located
    base_path = Path(__file__).parent.resolve()
    
    print("Starting cyclic compression data analysis...")
    
    # Create combined first cycle CSV file
    print(f"\n{'='*60}")
    print("Creating combined first cycle data CSV...")
    print(f"{'='*60}")
    first_cycle_csv_path = create_first_cycle_combined_csv(base_path)
    
    # Create histogram for first week first cycle peak forces if CSV exists
    if first_cycle_csv_path and Path('First_Week_First_Cycle_Combined_Data.csv').exists():
        print(f"\n{'='*60}")
        print("Creating first week first cycle peak force histogram...")
        print(f"{'='*60}")
        histogram_path = plot_first_cycle_peak_force_histogram('First_Week_First_Cycle_Combined_Data.csv')
        
        print(f"\n{'='*60}")
        print("Creating first week first cycle vertical stiffness histogram...")
        print(f"{'='*60}")
        vs_histogram_path = plot_first_cycle_vertical_stiffness_histogram('First_Week_First_Cycle_Combined_Data.csv')
    
    # Define all test weeks to analyze
    test_weeks = ['0 WK', '1 WK Repeats', '2 WK Repeats', '3 WK Repeats']
    
    for week_folder in test_weeks:
        test_name = get_test_name(week_folder)
        print(f"\n{'='*60}")
        print(f"Analyzing {test_name} ({week_folder})...")
        print(f"{'='*60}")
        
        # Check if folder exists
        week_path = Path(base_path) / week_folder
        if not week_path.exists():
            print(f"Warning: {week_folder} folder not found, skipping...")
            continue
            
        print("Creating output directories...")
        
        # Create organized output directories
        output_base = create_output_directories(test_name)
        print(f"Output directories created in: {output_base}")
        
        print(f"Loading {week_folder} data...")
        
        # Load and analyze week data
        data_by_combination = analyze_week_data(base_path, week_folder)
        
        if not data_by_combination:
            print(f"No data found in {week_folder}, skipping...")
            continue
        
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
        
        # Generate the three specific plot types requested
        print("\nGenerating plots...")
        
        print(f"Plot 1: Force vs displacement for specific cycles (1st, 100th, 1000th) - {test_name}...")
        plot_specific_cycles_force_displacement(data_by_combination, test_name, [1, 100, 1000])
        
        print(f"Plot 2: Peak force vs cycle grouped by filament type (same SVF, different TPU types) - {test_name}...")
        plot_by_filament_type(data_by_combination, test_name)
        
        print(f"Plot 3: Peak force vs cycle grouped by SVF percentage (same TPU filament, different SVF) - {test_name}...")
        plot_by_svf_percentage(data_by_combination, test_name)
        
        # Generate normal vs reprint comparison plots for Test 1 only
        if test_name == 'Test 1':
            print(f"Plot 4: Normal vs Reprint peak force comparison - {test_name}...")
            plot_normal_vs_reprint_comparison(data_by_combination, test_name)
        
        print(f"\n{test_name} analysis complete! Plots saved in: {output_base}")
        print(f"Generated plots in:")
        print(f"  - {test_name} force_displacement_cycles/: Force-displacement cycle plots")
        print(f"  - {test_name} by_filament_type/: TPU filament comparison plots (same SVF, different filaments)")
        print(f"  - {test_name} by_svf_percentage/: SVF comparison plots (same TPU filament, different SVFs)")
        if test_name == 'Test 1':
            print(f"  - {test_name} normal_vs_reprint_comparison/: Normal vs Reprint peak force comparison plots")
    
    print(f"\n{'='*60}")
    print("All individual tests analysis complete!")
    print(f"{'='*60}")
    
    # Generate cross-test comparison plots
    print(f"\n{'='*60}")
    print("Generating cross-test comparison plots...")
    print(f"{'='*60}")
    
    # Plot peak force vs cycle for every filament-SVF combination across all tests
    plot_cross_test_peak_force_comparison(base_path)
    
    # Plot specific cycles for every filament-SVF combination across all tests
    plot_cross_test_cycle_comparison(base_path, [1, 100, 1000])
    
    # Plot vertical stiffness comparison across all tests
    plot_cross_test_vs_comparison(base_path)
    
    print(f"\n{'='*60}")
    print("ALL ANALYSIS COMPLETE!")
    print("Generated:")
    print("1. Combined first cycle data CSV file (All_Tests_First_Cycle_Combined_Data.csv)")
    print("2. First week first cycle peak force histogram with RSD overlay")
    print("3. Individual test plots for each test week")
    print("4. Cross-test peak force comparison plots for each filament-SVF combination")
    print("5. Cross-test cycle comparison plots showing all combinations for cycles 1, 100, 1000")
    print("6. Cross-test vertical stiffness comparison plots for similar VS groups (within 40 N/mm)")
    if first_cycle_csv_path:
        print(f"\nFirst cycle combined CSV saved at: {first_cycle_csv_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to create only the first cycle CSV
    if len(sys.argv) > 1 and sys.argv[1] == '--first-cycle-only':
        # Set the base path to the directory where this script is located
        base_path = Path(__file__).parent.resolve()
        
        print("Creating combined first cycle CSV file only...")
        print(f"{'='*60}")
        first_cycle_csv_path = create_first_cycle_combined_csv(base_path)
        
        if first_cycle_csv_path:
            print(f"{'='*60}")
            print("FIRST CYCLE CSV CREATION COMPLETE!")
            print(f"Output file: {first_cycle_csv_path}")
            print(f"{'='*60}")
        else:
            print("Failed to create first cycle CSV file.")
    else:
        # Run the full analysis
        main()