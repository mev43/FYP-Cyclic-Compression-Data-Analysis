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
    Calculate vertical stiffness from the first-cycle force-displacement data.
    Returns slope (N/mm) or None if insufficient data.
    """
    try:
        first_cycle = df[df['Total Cycles'] == 1]
        if first_cycle.empty:
            return None
        force = first_cycle['Force(Linear:Load) (kN)'].values  # kN
        disp = first_cycle['Displacement(Linear:Digital Position) (mm)'].values  # mm
        if len(force) < 3 or len(disp) < 3:
            return None
        # Use linear regression on loading segment up to max |force|
        order = np.argsort(disp)
        disp = disp[order]
        force = force[order]
        end_idx = max(2, int(np.argmax(np.abs(force))))
        x = disp[:end_idx+1]
        y = force[:end_idx+1] * 1000.0  # convert to N
        if len(x) < 2:
            return None
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(slope)
    except Exception:
        return None

def find_peak_force_per_cycle(df):
    """Return (cycles_array, peak_force_array) excluding cycle 1001 if present."""
    try:
        df_clean = df[df['Total Cycles'] != 1001]
        cycles = sorted(df_clean['Total Cycles'].unique())
        peak_forces = []
        out_cycles = []
        for c in cycles:
            grp = df_clean[df_clean['Total Cycles'] == c]
            if grp.empty:
                continue
            peak = np.max(np.abs(grp['Force(Linear:Load) (kN)'].values))
            out_cycles.append(c)
            peak_forces.append(peak)
        return np.array(out_cycles), np.array(peak_forces)
    except Exception:
        return np.array([]), np.array([])

def load_tracking_data(file_path):
    """
    Load a tracking CSV in either legacy or Peak-Valley format and map to canonical columns.
    Returns a DataFrame or None on failure.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        legacy_cols = {
            'Total Cycles',
            'Displacement(Linear:Digital Position) (mm)',
            'Force(Linear:Load) (kN)'
        }
        if legacy_cols.issubset(set(df.columns)):
            return df
        # Attempt Peak-Valley parse: skip header (2 rows), provide names
        colnames = ['CycleCount', 'Axial Displacement', 'Axial Force']
        df2 = pd.read_csv(file_path, skiprows=2, names=colnames, encoding='utf-8-sig')
        # Coerce to numeric and drop NaNs
        df2['CycleCount'] = pd.to_numeric(df2['CycleCount'], errors='coerce')
        df2['Axial Displacement'] = pd.to_numeric(df2['Axial Displacement'], errors='coerce')
        df2['Axial Force'] = pd.to_numeric(df2['Axial Force'], errors='coerce')
        df2 = df2.dropna(subset=['CycleCount', 'Axial Displacement', 'Axial Force'])
        mapped = pd.DataFrame({
            'Total Cycles': df2['CycleCount'].astype(int),
            'Displacement(Linear:Digital Position) (mm)': df2['Axial Displacement'].astype(float),
            'Force(Linear:Load) (kN)': (df2['Axial Force'].astype(float)) / 1000.0
        })
        return mapped
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def find_tracking_csv(folder_path: Path) -> Path | None:
    """
    Find a tracking CSV file within a specimen folder.
    Preference order:
      1) Legacy path: Test1/Test1.steps.tracking.csv
      2) Any CSV containing 'Peak-Valley' in the name (recursively), most recent file if multiple
      3) Any '*.steps.tracking.csv' found recursively
    Returns the Path or None if not found.
    """
    # 1) Preferred legacy location
    legacy = folder_path / 'Test1' / 'Test1.steps.tracking.csv'
    if legacy.exists():
        return legacy

    # 2) New Peak-Valley format (recursively search)
    peak_valley_candidates = []
    try:
        for p in folder_path.rglob('*.csv'):
            name_lower = p.name.lower()
            if 'peak-valley' in name_lower:
                peak_valley_candidates.append(p)
    except Exception:
        peak_valley_candidates = []

    if peak_valley_candidates:
        # Pick the most recent modified file to be safe
        try:
            peak_valley_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass
        return peak_valley_candidates[0]

    # 3) Any steps.tracking.csv found recursively
    tracking_candidates = []
    try:
        for p in folder_path.rglob('*.steps.tracking.csv'):
            tracking_candidates.append(p)
    except Exception:
        tracking_candidates = []

    if tracking_candidates:
        try:
            tracking_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass
        return tracking_candidates[0]

    # 4) Any CSV that appears to be Peak-Valley by content (header contains quoted CycleCount & Axial Displacement & Axial Force)
    try:
        content_candidates = []
        for p in folder_path.rglob('*.csv'):
            try:
                with open(p, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    # Read a small chunk to inspect header area
                    preview = ''.join([next(f) for _ in range(12)])
                # Look for quoted column names (handle trailing spaces)
                if ('"CycleCount"' in preview and 'Axial Displacement' in preview and 'Axial Force' in preview):
                    content_candidates.append(p)
            except StopIteration:
                # Very short file; still check what we have
                try:
                    with open(p, 'r', encoding='utf-8-sig', errors='ignore') as f:
                        preview = f.read(512)
                    if ('"CycleCount"' in preview and 'Axial Displacement' in preview and 'Axial Force' in preview):
                        content_candidates.append(p)
                except Exception:
                    continue
            except Exception:
                continue
        if content_candidates:
            try:
                content_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            except Exception:
                pass
            return content_candidates[0]
    except Exception:
        pass

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
                
                # Locate tracking CSV file (supports legacy and new Peak-Valley formats)
                tracking_file = find_tracking_csv(folder)
                if tracking_file and tracking_file.exists():
                    print(f"  Using tracking file for {folder.name}: {tracking_file.relative_to(Path(base_path)) if str(tracking_file).startswith(str(base_path)) else tracking_file}")
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
    First averages VS values for the same filament-SVF combination, then groups by similarity
    """
    # First, group by filament-SVF combination and average their VS values
    combination_vs_averages = {}
    
    for combination_key, combo_data in data_by_combination.items():
        filament = combo_data['filament']
        svf = combo_data['svf']
        vs = combo_data['vs']
        
        # Create a key based on filament and SVF only (ignore individual VS differences)
        combo_key = (filament, svf)
        
        if combo_key not in combination_vs_averages:
            combination_vs_averages[combo_key] = []
        
        combination_vs_averages[combo_key].append(vs)
    
    # Calculate average VS for each filament-SVF combination
    averaged_combinations = {}
    for combo_key, vs_values in combination_vs_averages.items():
        filament, svf = combo_key
        avg_vs = np.mean(vs_values)
        
        # Store the averaged data
        averaged_combinations[combo_key] = {
            'filament': filament,
            'svf': svf,
            'avg_vs': avg_vs,
            'original_vs_values': vs_values,
            'original_combinations': []  # Store original combination keys that contributed to this average
        }
        
        # Find all original combinations that belong to this filament-SVF pair
        for orig_combination_key, orig_combo_data in data_by_combination.items():
            if orig_combo_data['filament'] == filament and orig_combo_data['svf'] == svf:
                averaged_combinations[combo_key]['original_combinations'].append(orig_combination_key)
    
    # Now group by similar averaged VS values
    avg_vs_values = [data['avg_vs'] for data in averaged_combinations.values()]
    avg_vs_values = sorted(set(avg_vs_values))  # Get unique averaged VS values, sorted
    
    vs_groups = []
    used_vs = set()
    
    for vs in avg_vs_values:
        if vs in used_vs:
            continue
            
        # Find all averaged VS values within tolerance
        group = [vs]
        used_vs.add(vs)
        
        for other_vs in avg_vs_values:
            if other_vs != vs and other_vs not in used_vs:
                if abs(vs - other_vs) <= tolerance:
                    group.append(other_vs)
                    used_vs.add(other_vs)
        
        vs_groups.append(group)
    
    return vs_groups, averaged_combinations

def plot_vs_groups_peak_force_comparison(data_by_combination, vs_groups, averaged_combinations, test_name):
    """
    Plot average peak force vs cycle for grouped vertical stiffnesses (within 40 N/mm)
    Now works with averaged vertical stiffness values for the same filament-SVF combinations
    and creates plots that should go in the vertical_stiffness_comparison folder
    """
    print(f"\nGenerating VS groups peak force comparison plots for {test_name}...")
    
    for group_idx, vs_group in enumerate(vs_groups):
        # Find which averaged combinations belong to this VS group
        combinations_in_group = []
        for avg_vs in vs_group:
            for combo_key, combo_data in averaged_combinations.items():
                if abs(combo_data['avg_vs'] - avg_vs) < 0.1:  # Small tolerance for floating point comparison
                    filament, svf = combo_data['filament'], combo_data['svf']
                    combinations_in_group.append((filament, svf, avg_vs))
        
        # Check if we have different combinations (different filament OR SVF)
        unique_combinations = set((f, s) for f, s, v in combinations_in_group)
        if len(unique_combinations) <= 1:
            print(f"  Skipping VS Group {group_idx+1} - only contains identical filament-SVF combinations")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_combinations)))
        
        # Sort combinations for consistent plotting
        unique_combinations = sorted(unique_combinations)
        
        for idx, (filament, svf) in enumerate(unique_combinations):
            color = colors[idx]
            filament_name = get_filament_name(filament)
            
            # Find all original combinations that belong to this filament-SVF pair
            all_forces_data = []
            
            # Look through original data to find matching combinations
            for combination_key, combo_data in data_by_combination.items():
                if combo_data['filament'] == filament and combo_data['svf'] == svf:
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
                
                # Get the averaged VS for this combination
                avg_vs = None
                for combo_key, combo_data in averaged_combinations.items():
                    if combo_data['filament'] == filament and combo_data['svf'] == svf:
                        avg_vs = combo_data['avg_vs']
                        break
                
                # Create label
                label = f'{filament_name} SVF {svf}% (VS = {avg_vs:.0f} N/mm, n={len(all_forces_data)})'
                
                # Plot average with error bands
                ax.plot(common_cycles, avg_forces, '-', color=color, 
                       label=label, linewidth=2, alpha=0.8)
                
                # Add standard deviation bands if multiple tests
                if len(interpolated_forces) > 1:
                    ax.fill_between(common_cycles, 
                                  avg_forces - std_forces/2, 
                                  avg_forces + std_forces/2, 
                                  color=color, alpha=0.2)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Cycle Number', fontsize=14)
        ax.set_ylabel('Average Peak Force (kN)', fontsize=14)
        
        # Create title with VS range and combination info
        vs_range = f"{min(vs_group):.0f}-{max(vs_group):.0f}" if len(vs_group) > 1 else f"{vs_group[0]:.0f}"
        combo_summary = ", ".join([f"{get_filament_name(f)} SVF {s}%" for f, s in unique_combinations])
        ax.set_title(f'{test_name} Peak Force vs Cycle (Averaged over two sets)\nSimilar Vertical Stiffnesses: VS Group {vs_range} N/mm\nCombinations: {combo_summary}', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add VS group annotation
        ax.text(0.02, 0.98, f'VS Group {group_idx+1}\n{vs_range} N/mm', 
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
               verticalalignment='top', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot in the vertical_stiffness_comparison folder
        output_path = Path(f'{test_name} analysis_results/{test_name} vertical_stiffness_comparison/{test_name} VS_Group_{group_idx+1}_{vs_range}_Nmm_peak_force_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  VS Group {group_idx+1} ({vs_range} N/mm) peak force comparison saved as: {output_path}")
    
    print(f"  Completed VS groups peak force comparison plots for {test_name}")

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
            # if disp_range > 0.1:  # Avoid division by zero
            #     target_range = 6.0  # Target 6mm compression range
            #     scale_factor = target_range / disp_range
            #     disp_normalized = disp_normalized * scale_factor
            
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
                    
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Displacement (mm)', fontsize=14)
        ax.set_ylabel('Force (kN)', fontsize=14)
        filament_name = get_filament_name(filament)
        # ax.set_title(f'{test_name} Force vs Displacement (Averaged over two sets) - {filament_name}, SVF {svf}%\n(Cycles 1, 100, 1000)')
        ax.legend(fontsize=14)
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
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
                    
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Cycle Number', fontsize=14)
        ax.set_ylabel('Average Peak Force (kN)', fontsize=14)
        # ax.set_title(f'{test_name} Peak Force vs Cycle (Averaged across two sets) - SVF {svf}%')
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(True, alpha=0.3)
        
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

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Cycle Number', fontsize=14)
        ax.set_ylabel('Average Peak Force (kN)', fontsize=14)
        filament_name = get_filament_name(filament)
        # ax.set_title(f'{test_name} Peak Force vs Cycle (Averaged over two sets) - {filament_name}\n(Comparison across SVF Percentages)')
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # # Add filament annotation
        # ax.text(0.02, 0.98, f'{filament_name}', transform=ax.transAxes, 
        #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9),
        #         verticalalignment='top', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filament_name_file = get_filament_name(filament)
        output_path = Path(f'{test_name} analysis_results/{test_name} by_svf_percentage/{test_name} {filament_name_file}_SVF_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SVF comparison plot for {filament_name} saved as: {output_path}")

def plot_specific_cycles_by_vs_groups(data_by_combination, vs_groups, averaged_combinations, test_name, target_cycles=[1, 100, 1000]):
    """
    Plot force vs displacement for specific cycles (1st, 100th, 1000th) 
    with separate plots for each VS group and each cycle.
    Each VS group gets its own folder with plots showing individual combination curves.
    
    Now uses averaged vertical stiffness values for the same filament-SVF combinations
    before grouping by similarity, so multiple specimens of the same combination are
    averaged together before plotting.
    """
    print(f"\nStarting VS groups plotting for {test_name}...")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Number of VS groups: {len(vs_groups)}")
    
    # Test base directory creation
    base_test_dir = Path(f'{test_name} analysis_results')
    vs_base_dir = base_test_dir / f'{test_name} force_displacement_by_vs_groups'
    
    print(f"Base test directory: {base_test_dir.absolute()}")
    print(f"VS base directory: {vs_base_dir.absolute()}")
    
    try:
        vs_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Successfully created base VS directory: {vs_base_dir}")
    except Exception as e:
        print(f"❌ Failed to create base VS directory: {e}")
        return
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
            # if disp_range > 0.1:  # Avoid division by zero
            #     target_range = 6.0  # Target 6mm compression range
            #     scale_factor = target_range / disp_range
            #     disp_normalized = disp_normalized * scale_factor
            
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
        for avg_vs in vs_group:
            # Find averaged combinations that match this VS value
            for combo_key, avg_data in averaged_combinations.items():
                if abs(avg_data['avg_vs'] - avg_vs) < 0.001:  # Match averaged VS values
                    filament, svf = avg_data['filament'], avg_data['svf']
                    combinations_in_group.append((filament, svf, avg_vs))
        
        # Check if we have different combinations (not just same filament-SVF with different VS)
        unique_combinations = set((f, s) for f, s, v in combinations_in_group)
        if len(unique_combinations) <= 1:
            print(f"Skipping VS Group {group_idx+1} - only contains identical filament-SVF combinations")
            continue
        
        # Create VS group folder
        vs_range = f"{min(vs_group):.0f}-{max(vs_group):.0f}" if len(vs_group) > 1 else f"{vs_group[0]:.0f}"
        # Clean up the folder name to avoid any path issues
        clean_test_name = test_name.replace(" ", "_")
        folder_name = f'{clean_test_name}_VS_Group_{group_idx+1}_{vs_range}_Nmm'
        group_folder = Path(f'{test_name} analysis_results') / f'{test_name} force_displacement_by_vs_groups' / folder_name
        
        # Ensure the directory exists before trying to save files
        try:
            group_folder.mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {group_folder}")
            print(f"  Directory exists: {group_folder.exists()}")
            print(f"  Current working directory: {Path.cwd()}")
        except Exception as e:
            print(f"  Error creating directory {group_folder}: {e}")
            continue
        
        for target_cycle in target_cycles:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create colors for each unique combination in this VS group
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_combinations)))
            combination_colors = {combo: colors[i] for i, combo in enumerate(sorted(unique_combinations))}
            
            plot_data_exists = False
            # Plot each unique combination separately (now averaged by filament-SVF)
            for avg_vs in vs_group:
                # Find averaged combinations that match this VS value
                for combo_key, avg_data in averaged_combinations.items():
                    if abs(avg_data['avg_vs'] - avg_vs) < 0.001:  # Match averaged VS values
                        filament, svf = avg_data['filament'], avg_data['svf']
                        filament_name = get_filament_name(filament)
                        combo_tuple = (filament, svf)
                        
                        if combo_tuple in combination_colors:
                            color = combination_colors[combo_tuple]
                            
                            # Collect data from all original combinations that contribute to this average
                            all_displacements_list = []
                            all_forces_list = []
                            
                            for orig_combination_key in avg_data['original_combinations']:
                                if orig_combination_key in data_by_combination:
                                    orig_combo_data = data_by_combination[orig_combination_key]
                                    
                                    # Get normal test data
                                    if 'normal' in orig_combo_data and orig_combo_data['normal'] and 'full_data' in orig_combo_data['normal']:
                                        normal_df = orig_combo_data['normal']['full_data']
                                        cycle_data = normal_df[normal_df['Total Cycles'] == target_cycle]
                                        if not cycle_data.empty:
                                            all_displacements_list.append(cycle_data['Displacement(Linear:Digital Position) (mm)'].values)
                                            all_forces_list.append(cycle_data['Force(Linear:Load) (kN)'].values)
                                    
                                    # Get reprint test data
                                    if 'reprint' in orig_combo_data and orig_combo_data['reprint'] and 'full_data' in orig_combo_data['reprint']:
                                        reprint_df = orig_combo_data['reprint']['full_data']
                                        cycle_data = reprint_df[reprint_df['Total Cycles'] == target_cycle]
                                        if not cycle_data.empty:
                                            all_displacements_list.append(cycle_data['Displacement(Linear:Digital Position) (mm)'].values)
                                            all_forces_list.append(cycle_data['Force(Linear:Load) (kN)'].values)
                            
                            # Calculate and plot average force-displacement curve for this averaged combination
                            if all_displacements_list and all_forces_list:
                                avg_result = average_hysteresis_loops(all_displacements_list, all_forces_list)
                                if avg_result is not None:
                                    avg_disp, avg_force = avg_result
                                    
                                    # Plot the averaged curve
                                    ax.plot(avg_disp, avg_force, color=color, linewidth=2.5, alpha=0.8,
                                           label=f'{filament_name} SVF {svf}% (Avg VS: {avg_vs:.0f} N/mm)')
                                    plot_data_exists = True
            
            # Save plot after processing all combinations for this cycle
            if plot_data_exists:
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_xlabel('Displacement (mm)', fontsize=14)
                ax.set_ylabel('Force (kN)', fontsize=14)
                combo_summary = ", ".join([f"{get_filament_name(f)}-SVF {s}%" for f, s in sorted(unique_combinations)])
                # ax.set_title(f'{test_name} Force vs Displacement - Cycle {target_cycle}\nVS Group {group_idx+1}: {vs_range} N/mm (Averaged by Filament-SVF)\nCombinations: {combo_summary}', 
                #             fontsize=14)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-7, 1)  # Show compression range
                
                # Add group info text box
                # info_text = f'VS Group {group_idx+1}\nRange: {vs_range} N/mm\nCombinations: {len(unique_combinations)}'
                # ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                #        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                #        verticalalignment='top', fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                
                # Save plot in the VS group folder
                clean_filename = f'{clean_test_name}_VS_Group_{group_idx+1}_Cycle_{target_cycle}_averaged_combinations.png'
                output_path = group_folder / clean_filename
                
                # Ensure the parent directory exists before saving
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"    Attempting to save to: {output_path}")
                print(f"    Parent directory exists: {output_path.parent.exists()}")
                print(f"    Full path: {output_path.absolute()}")
                
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"    ✅ VS Group {group_idx+1} Cycle {target_cycle} (averaged combinations) saved as: {output_path}")
                except Exception as e:
                    print(f"    ❌ Error saving plot {output_path}: {e}")
                    plt.close()
            else:
                plt.close()
                print(f"    No data found for VS Group {group_idx+1} Cycle {target_cycle}")
        
        print(f"  VS Group {group_idx+1} ({vs_range} N/mm) plots saved in: {group_folder}")

def plot_pairwise_force_displacement_overlaid(data_by_combination, test_name, pair_list, cycles_of_interest=[1, 100, 1000]):
    """
    Create one force vs displacement figure per requested pairwise comparison, overlaying
    the requested cycles on the same axes. Each test set (normal/reprint) is averaged per
    combination and then across sets. Replicates the style of the VS-group force-displacement
    plots but focused on specific pairwise comparisons instead of VS groups.
    """
    # Reuse averaging helper (time-based interpolation + normalization) from VS-group plots
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

            # Normalize displacement: zero at max (top), scale to ~ -6mm span if needed
            min_disp = np.min(disp_clean)
            max_disp = np.max(disp_clean)
            disp_normalized = disp_clean - max_disp

            disp_range = abs(min_disp - max_disp)
            # if disp_range > 0.1:
            #     target_range = 6.0
            #     scale_factor = target_range / disp_range
            #     disp_normalized = disp_normalized * scale_factor

            processed_data.append((disp_normalized, force_clean))

        if len(processed_data) < 1:
            return None, None

        if len(processed_data) == 1:
            return processed_data[0]

        # Time-based averaging on the longest series
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

    # Helper: collect all loops (normal + reprint across VS buckets) for a specific combination and cycle
    def collect_combo_cycle_loops(filament, svf, target_cycle):
        disps, forces = [] , []
        for _, combo_data in data_by_combination.items():
            if combo_data.get('filament') != filament or combo_data.get('svf') != svf:
                continue
            # Normal
            if 'normal' in combo_data and combo_data['normal'] and 'full_data' in combo_data['normal']:
                df = combo_data['normal']['full_data']
                cy = df[df['Total Cycles'] == target_cycle]
                if not cy.empty:
                    d = cy['Displacement(Linear:Digital Position) (mm)'].values
                    f = cy['Force(Linear:Load) (kN)'].values
                    if len(d) > 5 and len(f) > 5:
                        disps.append(d)
                        forces.append(f)
            # Reprint
            if 'reprint' in combo_data and combo_data['reprint'] and 'full_data' in combo_data['reprint']:
                df = combo_data['reprint']['full_data']
                cy = df[df['Total Cycles'] == target_cycle]
                if not cy.empty:
                    d = cy['Displacement(Linear:Digital Position) (mm)'].values
                    f = cy['Force(Linear:Load) (kN)'].values
                    if len(d) > 5 and len(f) > 5:
                        disps.append(d)
                        forces.append(f)
        return disps, forces

    # Output directory
    out_dir = Path(f'{test_name} analysis_results/{test_name} pairwise_force_displacement')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Line styles per cycle; no markers
    cycle_styles = {
        1:    {'linestyle': '-',  'linewidth': 2.6},
        100:  {'linestyle': '--', 'linewidth': 2.6},
        1000: {'linestyle': ':',  'linewidth': 2.6},
    }

    # Colors per combination in a pair
    combo_colors = [('tab:blue', 'tab:orange'), ('tab:green', 'tab:red'), ('tab:purple', 'tab:brown')]

    for idx, ((f1, s1), (f2, s2)) in enumerate(pair_list):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        colorA, colorB = combo_colors[idx % len(combo_colors)]
        labelA = f"{get_filament_name(f1)} SVF {s1}%"
        labelB = f"{get_filament_name(f2)} SVF {s2}%"

        plotted_any = False

        for cyc in cycles_of_interest:
            # Combo A
            dA, fA = collect_combo_cycle_loops(f1, s1, cyc)
            avg_dA, avg_fA = average_hysteresis_loops(dA, fA) if dA and fA else (None, None)
            # Combo B
            dB, fB = collect_combo_cycle_loops(f2, s2, cyc)
            avg_dB, avg_fB = average_hysteresis_loops(dB, fB) if dB and fB else (None, None)

            style = cycle_styles.get(cyc, {'linestyle': '-', 'linewidth': 2.5})

            if avg_dA is not None and avg_fA is not None:
                ax.plot(avg_dA, avg_fA, color=colorA, linestyle=style['linestyle'], linewidth=style['linewidth'],
                        alpha=0.9, label=f"{labelA} — Cycle {cyc}")
                plotted_any = True
            if avg_dB is not None and avg_fB is not None:
                ax.plot(avg_dB, avg_fB, color=colorB, linestyle=style['linestyle'], linewidth=style['linewidth'],
                        alpha=0.9, label=f"{labelB} — Cycle {cyc}")
                plotted_any = True

        if plotted_any:
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel('Displacement (mm)', fontsize=14)
            ax.set_ylabel('Force (kN)', fontsize=14)
            # ax.set_title(
            #     f"{test_name} Pairwise Force–Displacement\n{labelA} vs {labelB} — Cycles {', '.join(map(str, cycles_of_interest))}",
            #     fontsize=14, fontweight='bold'
            # )
            # De-duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=14, loc='best', ncol=1)
            ax.grid(True, alpha=0.3)
            # ax.text(0.02, 0.98, f"{labelA.split(' SVF')[0]} vs {labelB.split(' SVF')[0]}", transform=ax.transAxes,
            #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
            #         verticalalignment='top', fontsize=10, fontweight='bold')
            ax.set_xlim(-7, 1)
            plt.tight_layout()

            fname = f"{test_name}_{get_filament_name(f1)}_SVF_{s1}_vs_{get_filament_name(f2)}_SVF_{s2}_force_displacement_overlaid_cycles.png"
            out_path = out_dir / fname
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Pairwise FD (overlaid cycles) saved: {out_path}")
        else:
            plt.close()
            print(f"  Skipping pair {labelA} vs {labelB} — no usable data for requested cycles.")

def plot_pairwise_peak_force_vs_cycle(data_by_combination, test_name, pair_list):
    """
    Create pairwise average peak force vs cycle plots for specified comparisons.
    Each curve is averaged across normal + reprint and across VS buckets for that (filament, svf).
    Saves outputs to the vertical_stiffness_comparison folder with filenames like:
    Test_1_TPU87A_SVF_20_vs_TPU90A_SVF_20_peak_force_vs_cycle.png
    """
    def average_series(series_list):
        if not series_list:
            return None, None
        if len(series_list) == 1:
            return series_list[0]
        # Prefer intersection of cycles
        common = None
        for c, _ in series_list:
            s = set(c.tolist())
            common = s if common is None else (common & s)
        if common:
            common = np.array(sorted(list(common)))
            aligned = []
            for c, f in series_list:
                idx_map = {int(cc): i for i, cc in enumerate(c)}
                vals = []
                for cc in common:
                    i = idx_map.get(int(cc))
                    if i is None:
                        vals = None
                        break
                    vals.append(f[i])
                if vals is not None:
                    aligned.append(np.array(vals))
            if aligned:
                return common, np.mean(aligned, axis=0)
        # Fallback to min-length truncation
        min_len = min(len(c) for c, _ in series_list)
        if min_len == 0:
            return None, None
        cycles_ref = series_list[0][0][:min_len]
        avg_forces = np.mean([f[:min_len] for _, f in series_list], axis=0)
        return cycles_ref, avg_forces

    # Build grouped peak-force series by (filament, svf)
    grouped = {}
    for _, combo_data in data_by_combination.items():
        filament = combo_data.get('filament')
        svf = combo_data.get('svf')
        if filament is None or svf is None:
            continue
        key = (filament, svf)
        if key not in grouped:
            grouped[key] = []
        for set_name in ['normal', 'reprint']:
            if set_name in combo_data and combo_data[set_name]:
                set_data = combo_data[set_name]
                try:
                    if 'cycles' in set_data and 'peak_forces' in set_data and len(set_data['cycles']) and len(set_data['peak_forces']):
                        c_arr = np.array(set_data['cycles'])
                        f_arr = np.array(set_data['peak_forces'])
                    elif 'full_data' in set_data and set_data['full_data'] is not None:
                        c_arr, f_arr = find_peak_force_per_cycle(set_data['full_data'])
                    else:
                        c_arr, f_arr = np.array([]), np.array([])
                except Exception:
                    c_arr, f_arr = np.array([]), np.array([])
                if c_arr.size > 0 and f_arr.size > 0:
                    # Exclude cycle 1001 globally
                    mask = c_arr != 1001
                    c_arr = c_arr[mask]
                    f_arr = f_arr[mask]
                    order = np.argsort(c_arr)
                    grouped[key].append((c_arr[order], f_arr[order]))

    # Output directory
    out_dir = Path(f'{test_name} analysis_results/{test_name} vertical_stiffness_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    for ((f1, s1), (f2, s2)) in pair_list:
        sA = average_series(grouped.get((f1, s1), []))
        sB = average_series(grouped.get((f2, s2), []))
        if not sA or not sB or sA[0] is None or sB[0] is None:
            print(f"  Skipping pair {get_filament_name(f1)} SVF {s1}% vs {get_filament_name(f2)} SVF {s2}% — insufficient data.")
            continue

        c1, y1 = sA
        c2, y2 = sB
        # Align to common cycles for plotting
        common = sorted(list(set(c1.tolist()) & set(c2.tolist())))
        if common:
            common = np.array(common)
            idx1 = {int(cc): i for i, cc in enumerate(c1)}
            idx2 = {int(cc): i for i, cc in enumerate(c2)}
            y1p = np.array([y1[idx1[int(cc)]] for cc in common])
            y2p = np.array([y2[idx2[int(cc)]] for cc in common])
            cx = common
        else:
            min_len = min(len(c1), len(c2))
            cx = c1[:min_len]
            y1p = y1[:min_len]
            y2p = y2[:min_len]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        name1 = get_filament_name(f1)
        name2 = get_filament_name(f2)
        ax.plot(cx, y1p, linewidth=2.5, alpha=0.9, label=f"{name1} SVF {s1}%")
        ax.plot(cx, y2p, linewidth=2.5, alpha=0.9, label=f"{name2} SVF {s2}%")
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Cycle Number', fontsize=14)
        ax.set_ylabel('Average Peak Force (kN)', fontsize=14)
        # ax.set_title(f"{test_name} Pairwise Comparison\n{name1} SVF {s1}% vs {name2} SVF {s2}%", fontsize=14, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"{test_name}_{name1}_SVF_{s1}_vs_{name2}_SVF_{s2}_peak_force_vs_cycle.png"
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Pairwise peak force plot saved: {out_path}")

def plot_normal_vs_reprint_comparison(data_by_combination, test_name):
    """
    For each (filament, svf) in the specified test, plot peak force vs cycle comparing
    Normal vs Reprint, and save under:
      "{test_name} analysis_results/{test_name} normal_vs_reprint_comparison"
    """
    week_map = {'Test 1': '0 WK', 'Test 2': '1 WK Repeats', 'Test 3': '2 WK Repeats', 'Test 4': '3 WK Repeats'}
    base_path = Path(__file__).parent.resolve()
    week_folder = week_map.get(test_name, None)
    if week_folder is None:
        print(f"[Info] Unknown test name {test_name}; skipping normal vs reprint comparison.")
        return
    week_path = base_path / week_folder
    out_dir = Path(f"{test_name} analysis_results/{test_name} normal_vs_reprint_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Helper to compute peak force per cycle
    def pf_per_cycle(df):
        try:
            df2 = df[df['Total Cycles'] != 1001]
            cycles = sorted(df2['Total Cycles'].unique())
            out_c = []
            out_p = []
            for c in cycles:
                grp = df2[df2['Total Cycles'] == c]
                if grp.empty:
                    continue
                peak = float(np.max(np.abs(pd.to_numeric(grp['Force(Linear:Load) (kN)'], errors='coerce').to_numpy())))
                out_c.append(c)
                out_p.append(peak)
            return np.array(out_c), np.array(out_p)
        except Exception:
            return np.array([]), np.array([])

    # Gather series per combination and test type
    combos = {}
    if not week_path.exists():
        print(f"[Info] Week folder {week_folder} not found; skipping normal vs reprint comparison.")
        return
    for folder in week_path.iterdir():
        if not folder.is_dir():
            continue
        # Parse filament/SVF
        filament, svf = None, None
        try:
            filament, svf = extract_parameters_from_filename(folder.name)
            if filament is None or svf is None:
                filament, svf = _parse_filament_svf_from_name(folder.name)
        except Exception:
            filament, svf = _parse_filament_svf_from_name(folder.name)
        if filament is None or svf is None:
            continue
        ttype = 'reprint' if 'REPRINT' in folder.name.upper() else 'normal'
        fcsv = find_tracking_csv(folder)
        if not fcsv:
            continue
        df = load_tracking_data(fcsv)
        if df is None:
            continue
        c, p = pf_per_cycle(df)
        key = (filament, svf)
        if key not in combos:
            combos[key] = {}
        combos[key][ttype] = (c, p)

    # Plot per combo with both series
    for (f, s), series_map in combos.items():
        if 'normal' not in series_map or 'reprint' not in series_map:
            continue
        cn, pn = series_map['normal']
        cr, pr = series_map['reprint']
        if len(cn) == 0 or len(cr) == 0:
            continue
        common = sorted(list(set(cn.tolist()) & set(cr.tolist())))
        if common:
            pn_plot = np.array([pn[cn.tolist().index(c)] for c in common])
            pr_plot = np.array([pr[cr.tolist().index(c)] for c in common])
            cx = np.array(common)
        else:
            # Fallback: align by min length
            m = min(len(cn), len(cr))
            cx = cn[:m]
            pn_plot = pn[:m]
            pr_plot = pr[:m]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        name = get_filament_name(f)
        ax.plot(cx, pn_plot, label='Print 1', color='tab:blue', linewidth=2.3)
        ax.plot(cx, pr_plot, label='Print 2', color='tab:orange', linewidth=2.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Cycle Number', fontsize=14)
        ax.set_ylabel('Peak Force (kN)', fontsize=14)
        # ax.set_title(f'{test_name} Normal vs Reprint\n{name} SVF {s}%', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
        plt.tight_layout()
        fname = f"{test_name}_{name}_SVF_{s}_normal_vs_reprint_peak_force_vs_cycle.png"
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved normal vs reprint comparison: {out_path}")

def create_tpu95a_50_detailed_comparison(data_by_combination, test_name):
    """
    Placeholder: Intended to create a special detailed comparison for TPU95A 50%.
    Currently logs an informational message and skips plotting to avoid NameErrors.
    """
    try:
        print("[Info] TPU95A 50% detailed comparison skipped (not implemented).")
    except Exception as e:
        print(f"[Warn] create_tpu95a_50_detailed_comparison encountered an issue and was skipped: {e}")

def create_first_test_first_cycle_csv(base_path):
    """
    Create and return the CSV path for Test 1, cycle 1, by delegating to create_first_test_cycle_csv.
    Returns a pathlib.Path to the CSV if created, else None.
    """
    try:
        result = create_first_test_cycle_csv(base_path, 1)
        # Normalize to Path object when successful
        return Path(result) if result else None
    except Exception as e:
        print(f"[Warn] Failed to create first test first cycle CSV: {e}")
        return None

def plot_first_cycle_vertical_stiffness_histogram(csv_path):
    """
    Create a vertical stiffness histogram for cycle 1 using the First Test Cycle 1 CSV.
    Output: Test_1_First_Cycle_Vertical_Stiffness_Histogram.png
    """
    print("\nCreating first cycle vertical stiffness histogram...")
    try:
        df = pd.read_csv(csv_path)
        if 'Cycle_Number' in df.columns:
            df = df[df['Cycle_Number'] == 1]
        req = {'Filament_Name', 'SVF_Percentage', 'Test_Type',
               'Displacement(Linear:Digital Position) (mm)', 'Force(Linear:Load) (kN)'}
        if not req.issubset(set(df.columns)):
            print("  CSV missing required columns for VS computation; skipping.")
            return None

        def compute_vs_from_group(g):
            # Compute VS using the original acquisition order up to the maximum |force|
            x = pd.to_numeric(g['Displacement(Linear:Digital Position) (mm)'], errors='coerce').to_numpy()
            ykN = pd.to_numeric(g['Force(Linear:Load) (kN)'], errors='coerce').to_numpy()
            mask = np.isfinite(x) & np.isfinite(ykN)
            x = x[mask]
            y = (ykN[mask] * 1000.0)  # N
            if len(x) < 3:
                return np.nan
            end_idx = int(np.nanargmax(np.abs(y)))
            end_idx = max(2, end_idx)
            X = x[:end_idx+1]
            Y = y[:end_idx+1]
            if len(X) < 2:
                return np.nan
            A = np.vstack([X, np.ones(len(X))]).T
            slope, _ = np.linalg.lstsq(A, Y, rcond=None)[0]
            return float(slope)

        # Ensure specimen identity is available; otherwise, approximate with combined group label
        group_cols = ['Filament_Name', 'SVF_Percentage', 'Test_Type']
        if 'Folder_Name' in df.columns:
            group_cols.append('Folder_Name')

        # Compute per-specimen VS
        vs_records = []
        for keys, g in df.groupby(group_cols):
            if isinstance(keys, tuple):
                fil, svf, ttype = keys[0], keys[1], keys[2]
            else:
                fil, svf, ttype = keys, None, None  # Should not happen, safety only
            vs_val = compute_vs_from_group(g)
            rec = {'Filament_Name': fil, 'SVF_Percentage': svf, 'Test_Type': ttype, 'VS_N_per_mm': vs_val}
            if 'Folder_Name' in df.columns and isinstance(keys, tuple) and len(keys) >= 4:
                rec['Folder_Name'] = keys[3]
            vs_records.append(rec)
        vs_df = pd.DataFrame(vs_records)
        if vs_df.empty:
            print("  No VS values computed; skipping VS histogram.")
            return None

        stats_rows = []
        for (fil, svf), sub in vs_df.groupby(['Filament_Name', 'SVF_Percentage']):
            # Average within each test type across specimens
            norm_vals = sub.loc[sub['Test_Type'].str.lower() == 'normal', 'VS_N_per_mm'].dropna().values
            rep_vals = sub.loc[sub['Test_Type'].str.lower() == 'reprint', 'VS_N_per_mm'].dropna().values
            norm_val = float(np.mean(norm_vals)) if len(norm_vals) > 0 else np.nan
            rep_val = float(np.mean(rep_vals)) if len(rep_vals) > 0 else np.nan
            vals = [v for v in [norm_val, rep_val] if pd.notna(v)]
            mean_vs = float(np.mean(vals)) if vals else np.nan
            std_vs = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rsd = float((std_vs / mean_vs) * 100.0) if vals and mean_vs != 0 else 0.0
            stats_rows.append({
                'ComboLabel': f"{fil}\nSVF {int(svf)}%",
                'Normal_VS': norm_val if pd.notna(norm_val) else 0.0,
                'Reprint_VS': rep_val if pd.notna(rep_val) else 0.0,
                'Mean_VS': mean_vs,
                'Std_VS': std_vs,
                'RSD_Percent': rsd,
            })
        stats = pd.DataFrame(stats_rows)
        if stats.empty:
            print("  No per-combination VS stats; skipping plot.")
            return None

        x = np.arange(len(stats))
        width = 0.35
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
        ax1.bar(x - width/2, stats['Normal_VS'], width, label='Print 1', color='palegreen', edgecolor='green', alpha=0.85)
        ax1.bar(x + width/2, stats['Reprint_VS'], width, label='Print 2', color='moccasin', edgecolor='darkorange', alpha=0.85)
        # Plot mean ± SD without connecting lines between means (marker-only)
        # ax1.errorbar(x, stats['Mean_VS'], yerr=stats['Std_VS'], fmt='ok', capsize=5, label='Mean Vertical Stiffness')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel('Filament-SVF Combinations', fontsize=14)
        ax1.set_ylabel('Vertical Stiffness (N/mm)', fontsize=14)
        # ax1.set_title('Test 1 (First Week) First Cycle Vertical Stiffness Analysis\nVertical Stiffness by Combination with Individual RSD Calculations', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(stats['ComboLabel'], rotation=45, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.legend(loc='upper left', fontsize=14)
        ax2 = ax1.twinx()
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylabel('Relative Standard Deviation (%)', fontsize=14, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # Draw per-combination RSD chain computed from normal/reprint around their midpoint
        colors = plt.cm.Set3(np.linspace(0, 1, len(stats)))
        for i, (xi, rsd, n_val, r_val) in enumerate(zip(x, stats['RSD_Percent'], stats['Normal_VS'], stats['Reprint_VS'])):
            # Only draw chain if we have both values and the midpoint is nonzero
            if n_val > 0 and r_val > 0:
                mean_val = (n_val + r_val) / 2.0
                if mean_val != 0:
                    offset_n = ((n_val - mean_val) / mean_val) * 100.0
                    offset_r = ((r_val - mean_val) / mean_val) * 100.0
                    ax2.plot([xi - width/2, xi + width/2], [rsd + offset_n, rsd + offset_r], linewidth=2, alpha=0.8, c='black', marker='o')
            # Annotate RSD at center
            ax2.annotate(f"{rsd:.1f}%", (xi, rsd), textcoords="offset points", xytext=(0, 8),
                         ha='center', fontsize=14, color='red')
        from matplotlib.lines import Line2D
        # ax2.legend(handles=[Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='RSD Chain (Normal ↔ Reprint)')], loc='upper right', fontsize=14)
        plt.tight_layout()
        out_path = Path('Test_1_First_Cycle_Vertical_Stiffness_Histogram.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved first-cycle vertical stiffness histogram: {out_path}")
        return out_path
    except Exception as e:
        print(f"[Warn] plot_first_cycle_vertical_stiffness_histogram failed: {e}")
        return None

def plot_vertical_stiffness_at_specific_cycles(data_by_combination, test_name, cycles):
    """
    Generate vertical stiffness histograms for specified cycles for a given test week.
    Creates one grouped bar chart per cycle under:
      "{test_name} analysis_results/{test_name} vertical_stiffness_at_cycles"
    """
    week_map = {'Test 1': '0 WK', 'Test 2': '1 WK Repeats', 'Test 3': '2 WK Repeats', 'Test 4': '3 WK Repeats'}
    base_path = Path(__file__).parent.resolve()
    week_folder = week_map.get(test_name, None)
    if week_folder is None:
        print(f"[Info] Unknown test name {test_name}; skipping VS-at-cycles plots.")
        return
    week_path = base_path / week_folder
    out_dir = Path(f"{test_name} analysis_results/{test_name} vertical_stiffness_at_cycles")
    out_dir.mkdir(parents=True, exist_ok=True)

    def compute_vs_for_cycle(df, cyc):
        try:
            sub = df[df['Total Cycles'] == cyc]
            if sub.empty:
                return np.nan
            x = pd.to_numeric(sub['Displacement(Linear:Digital Position) (mm)'], errors='coerce').to_numpy()
            y = pd.to_numeric(sub['Force(Linear:Load) (kN)'], errors='coerce').to_numpy() * 1000.0
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(x) < 3:
                return np.nan
            # Use original order up to the point of maximum |force|
            end_idx = int(np.nanargmax(np.abs(y)))
            end_idx = max(2, end_idx)
            X = x[:end_idx+1]
            Y = y[:end_idx+1]
            if len(X) < 2:
                return np.nan
            A = np.vstack([X, np.ones(len(X))]).T
            slope, _ = np.linalg.lstsq(A, Y, rcond=None)[0]
            return float(slope)
        except Exception:
            return np.nan

    data = {c: {} for c in cycles}
    if not week_path.exists():
        print(f"[Info] Week folder {week_folder} not found; skipping VS-at-cycles plots.")
        return
    for folder in week_path.iterdir():
        if not folder.is_dir():
            continue
        filament, svf = None, None
        try:
            filament, svf = extract_parameters_from_filename(folder.name)
            if filament is None or svf is None:
                filament, svf = _parse_filament_svf_from_name(folder.name)
        except Exception:
            filament, svf = _parse_filament_svf_from_name(folder.name)
        if filament is None or svf is None:
            continue
        ttype = 'reprint' if 'REPRINT' in folder.name.upper() else 'normal'
        fcsv = find_tracking_csv(folder)
        if not fcsv:
            continue
        df = load_tracking_data(fcsv)
        if df is None:
            continue
        key = (filament, svf)
        for cyc in cycles:
            val = compute_vs_for_cycle(df, cyc)
            if np.isnan(val):
                continue
            if key not in data[cyc]:
                data[cyc][key] = {'normal': [], 'reprint': []}
            data[cyc][key][ttype].append(val)

    for cyc in cycles:
        combos = sorted(list(data[cyc].keys()))
        if not combos:
            print(f"  No VS data for cycle {cyc}; skipping.")
            continue
        normals, reprints, means, stds, rsds, labels = [], [], [], [], [], []
        for (f, s) in combos:
            vals_n = data[cyc][(f, s)]['normal']
            vals_r = data[cyc][(f, s)]['reprint']
            nmean = float(np.mean(vals_n)) if vals_n else 0.0
            rmean = float(np.mean(vals_r)) if vals_r else 0.0
            both = [v for v in [nmean, rmean] if v > 0]
            m = float(np.mean(both)) if both else 0.0
            sd = float(np.std(both, ddof=1)) if len(both) > 1 else 0.0
            rsd = float((sd / m) * 100.0) if m != 0 else 0.0
            normals.append(nmean)
            reprints.append(rmean)
            means.append(m)
            stds.append(sd)
            rsds.append(rsd)
            labels.append(f"{get_filament_name(f)}\nSVF {int(s)}%")

        x = np.arange(len(combos))
        width = 0.35
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
        ax1.bar(x - width/2, normals, width, label='Print 1', color='palegreen', edgecolor='green', alpha=0.85)
        ax1.bar(x + width/2, reprints, width, label='Print 2', color='moccasin', edgecolor='darkorange', alpha=0.85)
        # Plot mean ± SD without connecting lines between means (marker-only)
        # ax1.errorbar(x, means, yerr=stds, fmt='ok', capsize=5, label='Mean Vertical Stiffness')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel('Filament-SVF Combinations', fontsize=14)
        ax1.set_ylabel('Vertical Stiffness (N/mm)', fontsize=14)
        # ax1.set_title(f'{test_name} Vertical Stiffness at Cycle {cyc}', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.legend(loc='upper left', fontsize=14)
        ax2 = ax1.twinx()
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylabel('Relative Standard Deviation (%)', fontsize=14, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        colors = plt.cm.Set3(np.linspace(0, 1, len(x)))
        for i, (xi, rsd, n_val, r_val) in enumerate(zip(x, rsds, normals, reprints)):
            if n_val > 0 and r_val > 0:
                mean_val = (n_val + r_val) / 2.0
                if mean_val != 0:
                    offset_n = ((n_val - mean_val) / mean_val) * 100.0
                    offset_r = ((r_val - mean_val) / mean_val) * 100.0
                    ax2.plot([xi - width/2, xi + width/2], [rsd + offset_n, rsd + offset_r],
                             linewidth=2, alpha=0.8, c='black', marker='o')
            ax2.annotate(f"{rsd:.1f}%", (xi, rsd), textcoords="offset points", xytext=(0, 8),
                         ha='center', fontsize=14, color='red')
        from matplotlib.lines import Line2D
        # ax2.legend(handles=[Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='RSD Chain (Normal ↔ Reprint)')], loc='upper right', fontsize=14)
        plt.tight_layout()
        out_path = out_dir / f'Cycle_{cyc}_Vertical_Stiffness_Histogram.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved VS-at-cycle plot: {out_path}")

def plot_test1_vertical_stiffness_comparison_across_cycles(base_path, cycles=[1, 100, 1000]):
    """
    Replicate the 'Test 1 Vertical Stiffness Comparison Across Cycles' figure:
    - X: Filament–SVF combinations (TPUxxA SVF yy%)
    - Grouped bars per combo for cycles [1, 100, 1000]
    - Height = average vertical stiffness (N/mm) across Print 1 and Print 2 (all specimens)
    - Adds value labels on bars and a summary box with overall means per cycle

    Output: 'Test 1 analysis_results/Test 1 vertical_stiffness_at_cycles/Test_1_VS_Comparison_Across_Cycles.png'
    """
    try:
        week_folder = '0 WK'
        test_name = 'Test 1'
        week_path = Path(base_path) / week_folder
        if not week_path.exists():
            print(f"[Info] Week folder '{week_folder}' not found; skipping VS comparison across cycles plot.")
            return None

        # Helper: compute VS for a given cycle from a tracking dataframe
        def compute_vs_for_cycle(df, cyc):
            try:
                sub = df[df['Total Cycles'] == cyc]
                if sub.empty:
                    return None
                # Use original acquisition order up to max |force|
                disp = sub['Displacement(Linear:Digital Position) (mm)'].to_numpy()
                force_kN = sub['Force(Linear:Load) (kN)'].to_numpy()
                if len(disp) < 3 or len(force_kN) < 3:
                    return None
                end_idx = max(2, int(np.argmax(np.abs(force_kN))))
                x = disp[:end_idx + 1]
                yN = force_kN[:end_idx + 1] * 1000.0  # N
                if len(x) < 2:
                    return None
                A = np.vstack([x, np.ones(len(x))]).T
                slope, _ = np.linalg.lstsq(A, yN, rcond=None)[0]
                return float(slope)
            except Exception:
                return None

        # Collect VS per (filament, svf, cycle) per Test_Type, then average Normal & Reprint means
        data = {}  # {(filament, svf): {cycle: {'normal': [vs...], 'reprint': [vs...]}}}
        for folder in week_path.iterdir():
            if not folder.is_dir():
                continue
            # Parse filament & SVF robustly
            filament, svf = (None, None)
            try:
                filament, svf = _parse_filament_svf_from_name(folder.name)
            except Exception:
                pass
            if filament is None or svf is None:
                f2, s2 = extract_parameters_from_filename(folder.name)
                filament = filament if filament is not None else f2
                svf = svf if svf is not None else s2
            if filament is None or svf is None:
                continue

            fcsv = find_tracking_csv(folder)
            if not fcsv:
                continue
            df = load_tracking_data(fcsv)
            if df is None:
                continue

            key = (int(filament), int(svf))
            if key not in data:
                data[key] = {c: {'normal': [], 'reprint': []} for c in cycles}
            ttype = 'reprint' if 'REPRINT' in folder.name.upper() else 'normal'
            for cyc in cycles:
                vs = compute_vs_for_cycle(df, cyc)
                if vs is not None and np.isfinite(vs):
                    data[key][cyc][ttype].append(vs)

        if not data:
            print("[Info] No VS data found for Test 1; skipping VS comparison across cycles plot.")
            return None

        # Prepare plot arrays: one bar per cycle for each combo, values = average of (Normal mean, Reprint mean)
        combos = sorted(list(data.keys()))
        vals = {c: [] for c in cycles}
        for combo in combos:
            for c in cycles:
                bucket = data[combo].get(c, {'normal': [], 'reprint': []})
                n_arr = np.array(bucket.get('normal', []), dtype=float)
                r_arr = np.array(bucket.get('reprint', []), dtype=float)
                n_arr = n_arr[np.isfinite(n_arr)]
                r_arr = r_arr[np.isfinite(r_arr)]
                n_mean = float(np.mean(n_arr)) if n_arr.size > 0 else np.nan
                r_mean = float(np.mean(r_arr)) if r_arr.size > 0 else np.nan
                if np.isfinite(n_mean) and np.isfinite(r_mean):
                    vals[c].append((n_mean + r_mean) / 2.0)
                elif np.isfinite(n_mean):
                    vals[c].append(n_mean)
                elif np.isfinite(r_mean):
                    vals[c].append(r_mean)
                else:
                    vals[c].append(np.nan)

        # Figure and axes
        num_combos = len(combos)
        x = np.arange(num_combos)
        group_width = min(0.8, 0.15 + 0.1 * len(cycles))
        width = group_width / max(1, len(cycles))
        fig_width = max(16, 0.75 * num_combos + 6)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 9))

        color_map = {1: '#66b3ff', 100: '#ff9999', 1000: '#99ff99'}  # blue, red, green-ish
        bars_per_cycle = {}
        for idx, c in enumerate(cycles):
            offsets = -group_width/2 + (idx + 0.5) * width
            bar_x = x + offsets
            heights = vals[c]
            bars = ax.bar(bar_x, heights, width, label=f'Cycle {c}',
                          color=color_map.get(c, None), edgecolor='black', alpha=0.85)
            bars_per_cycle[c] = (bar_x, bars)

            # Value labels rounded to nearest integer
            # for bx, h in zip(bar_x, heights):
            #     if np.isfinite(h):
            #         ax.text(bx, h + max(2, 0.01 * (np.nanmax(heights) if np.nanmax(heights) > 0 else 1)),
            #                 f"{int(round(h))}", ha='center', va='bottom', fontsize=11, rotation=0)

        # Axes formatting
        combo_labels = [f"{get_filament_name(f)} SVF {s}%" for (f, s) in combos]
        ax.set_xticks(x)
        ax.set_xticklabels(combo_labels, rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Filament-SVF Combinations', fontsize=14)
        ax.set_ylabel('Average Vertical Stiffness (N/mm)', fontsize=14)
        # ax.set_title('Test 1 Vertical Stiffness Comparison Across Cycles\nAll Combinations (Average of Normal & Reprint)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=14)

        # Summary statistics box (overall mean across combinations for each cycle)
        try:
            lines = []
            for c in cycles:
                arr = np.array(vals[c], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    lines.append(f"Cycle {c}: {np.mean(arr):.0f}±{np.std(arr, ddof=1):.1f} N/mm")
                else:
                    lines.append(f"Cycle {c}: N/A")
            lines.append(f"Total Combinations: {len(combos)}")
            text = "Summary Statistics:\n" + "\n".join(lines)
            # ax.text(0.985, 0.97, text, transform=ax.transAxes, ha='right', va='top',
            #         bbox=dict(boxstyle='round,pad=0.5', facecolor='ivory', alpha=0.9), fontsize=10)
        except Exception:
            pass

        plt.tight_layout()
        out_dir = Path(f'{test_name} analysis_results/{test_name} vertical_stiffness_at_cycles')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'Test_1_VS_Comparison_Across_Cycles.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved VS comparison across cycles plot: {out_path}")
        return out_path
    except Exception as e:
        print(f"[Warn] plot_test1_vertical_stiffness_comparison_across_cycles failed: {e}")
        return None

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
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel('Cycle Number', fontsize=14)
            ax.set_ylabel('Average Peak Force (kN)', fontsize=14)
            # ax.set_title(f'All Tests Peak Force vs Cycle (Averaged across two sets)\n{filament_name}, SVF {svf}%', fontsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add combination annotation
            # ax.text(0.02, 0.98, f'{filament_name}\nSVF {svf}%', transform=ax.transAxes, 
            #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
            #         verticalalignment='top', fontsize=12)
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
            # if disp_range > 0.1:
            #     target_range = 6.0
            #     # scale_factor = target_range / disp_range
            #     # disp_normalized = disp_normalized * scale_factor
            
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
                                    label=f'{test_name}')
                            plot_created = True
            
            if plot_created:
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_xlabel('Displacement (mm)', fontsize=14)
                ax.set_ylabel('Force (kN)', fontsize=14)
                # ax.set_title(f'All Tests Force vs Displacement (Averaged across two sets) - Cycle {target_cycle}\n{filament_name}, SVF {svf}%',
                #              fontsize=14, fontweight='bold')
                ax.legend(fontsize=14, loc='best')
                ax.grid(True, alpha=0.3)
                
                # Add combination annotation
                # ax.text(0.02, 0.98, f'{filament_name}\nSVF {svf}%', transform=ax.transAxes, 
                #        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
                #        verticalalignment='top', fontsize=10, fontweight='bold')
                
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

def plot_cross_test_peak_force_by_filament(base_path):
    """
    Create three plots (one per filament: TPU87A, TPU90A, TPU95A) that overlay
    the peak force vs cycle curves for ALL SVFs for that filament, with one
    line per test (Test 1–4) for each SVF.

    Saves to: 'Cross Test Comparison analysis_results/Cross Test peak_force_vs_cycle'
    with filenames like 'Cross_Test_TPU87A_all_SVFs_peak_force_vs_cycle.png'.
    """
    print("\nGenerating cross-test peak force per filament plots...")
    week_map = {
        'Test 1': '0 WK',
        'Test 2': '1 WK Repeats',
        'Test 3': '2 WK Repeats',
        'Test 4': '3 WK Repeats',
    }

    # Helper: compute peak force per cycle for a dataframe
    def pf_per_cycle(df):
        try:
            df_clean = df[df['Total Cycles'] != 1001]
            cycles = sorted(df_clean['Total Cycles'].unique())
            out_c, out_p = [], []
            for c in cycles:
                sub = df_clean[df_clean['Total Cycles'] == c]
                if sub.empty:
                    continue
                peak_kN = np.max(np.abs(sub['Force(Linear:Load) (kN)'].to_numpy()))
                out_c.append(c)
                out_p.append(peak_kN)
            return np.array(out_c, dtype=int), np.array(out_p, dtype=float)
        except Exception:
            return np.array([], dtype=int), np.array([], dtype=float)

    # Helper: average a list of (cycles, values) series into one
    def average_series(series_list):
        if not series_list:
            return None, None
        if len(series_list) == 1:
            return series_list[0]
        # Prefer intersection of cycles
        common = None
        for c, _ in series_list:
            if common is None:
                common = set(c.tolist())
            else:
                common &= set(c.tolist())
        if common:
            cx = np.array(sorted(common), dtype=int)
            aligned = []
            for c, y in series_list:
                idx = np.searchsorted(c, cx)
                # Ensure exact match
                mask = (idx < len(c)) & (c[idx] == cx)
                vals = np.full_like(cx, np.nan, dtype=float)
                vals[mask] = y[idx[mask]]
                aligned.append(vals)
            Y = np.nanmean(np.stack(aligned, axis=0), axis=0)
            # Drop NaNs if any
            valid = np.isfinite(Y)
            return (cx[valid], Y[valid]) if np.any(valid) else (None, None)
        # Fallback to min-length truncation on first series order
        min_len = min(len(c) for c, _ in series_list)
        if min_len == 0:
            return None, None
        cx = series_list[0][0][:min_len]
        Y = np.mean([y[:min_len] for _, y in series_list], axis=0)
        return cx, Y

    # Gather series keyed by (filament, svf, test)
    grouped = {}  # {(filament, svf, test_name): [ (cycles, peaks), ... ]}
    for test_name, week_folder in week_map.items():
        week_path = Path(base_path) / week_folder
        if not week_path.exists():
            continue
        for folder in week_path.iterdir():
            if not folder.is_dir():
                continue
            # Parse filament & SVF
            filament, svf = (None, None)
            try:
                filament, svf = _parse_filament_svf_from_name(folder.name)
            except Exception:
                pass
            if filament is None or svf is None:
                f2, s2 = extract_parameters_from_filename(folder.name)
                filament = filament if filament is not None else f2
                svf = svf if svf is not None else s2
            if filament is None or svf is None:
                continue
            fcsv = find_tracking_csv(folder)
            if not fcsv:
                continue
            df = load_tracking_data(fcsv)
            if df is None:
                continue
            c, p = pf_per_cycle(df)
            if c.size == 0:
                continue
            key = (int(filament), int(svf), test_name)
            grouped.setdefault(key, []).append((c, p))

    # Average series per key
    averaged = {}
    for key, series_list in grouped.items():
        cx, y = average_series(series_list)
        if cx is None or y is None or len(cx) == 0:
            continue
        averaged[key] = (cx, y)

    # Arrange by filament -> SVF -> test
    filaments = sorted({k[0] for k in averaged.keys()})
    if not filaments:
        print("  No cross-test data available for per-filament plots; skipping.")
        return

    out_dir = Path('Cross Test Comparison analysis_results') / 'Cross Test peak_force_vs_cycle'
    out_dir.mkdir(parents=True, exist_ok=True)

    # High-contrast styles per test (distinct linestyles + markers)
    test_styles = {
        'Test 1': {'linestyle': '-',  'linewidth': 2.4, 'marker': 'o'},
        'Test 2': {'linestyle': '--', 'linewidth': 2.4, 'marker': 's'},
        'Test 3': {'linestyle': '-.', 'linewidth': 2.4, 'marker': 'D'},
        'Test 4': {'linestyle': ':',  'linewidth': 2.6, 'marker': '^'},
    }

    for fil in filaments:
        # Gather available SVFs for this filament
        svfs = sorted({k[1] for k in averaged.keys() if k[0] == fil})
        if not svfs:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        # Use a colorblind-friendly, high-contrast palette for SVFs
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(svfs))]
        svf_color = {svf: colors[i] for i, svf in enumerate(svfs)}

        plotted = False
        for svf in svfs:
            for test_name in ['Test 1', 'Test 2', 'Test 3', 'Test 4']:
                key = (fil, svf, test_name)
                if key not in averaged:
                    continue
                cx, y = averaged[key]
                style = test_styles.get(test_name, {'linestyle': '-', 'linewidth': 2.2, 'marker': None})
                # Reduce marker clutter with markevery; ensure at least every 1 if short series
                me = max(1, len(cx) // 20)
                ax.plot(
                    cx,
                    y,
                    color=svf_color[svf],
                    marker=style.get('marker', None),
                    markevery=me if style.get('marker') else None,
                    markersize=5.5,
                    linestyle=style.get('linestyle', '-'),
                    linewidth=style.get('linewidth', 2.2),
                    alpha=0.95,
                )
                plotted = True

        name = get_filament_name(fil)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Cycle Number', fontsize=14)
        ax.set_ylabel('Average Peak Force (kN)', fontsize=14)
        # ax.set_title(f'Cross Test: {name} — All SVFs\nPeak Force vs Cycle', fontsize=14, fontweight='bold')
        if plotted:
            # Build separate legends: one mapping color->SVF, another mapping linestyle/marker->Test
            from matplotlib.lines import Line2D
            svf_handles = [
                Line2D([0], [0], color=svf_color[s], lw=3, label=f'SVF {s}%')
                for s in svfs
            ]
            test_handles = [
                Line2D(
                    [0], [0],
                    color='black',
                    lw=test_styles[t]['linewidth'],
                    ls=test_styles[t]['linestyle'],
                    marker=test_styles[t]['marker'],
                    markersize=6,
                    label=t,
                )
                for t in ['Test 1', 'Test 2', 'Test 3', 'Test 4']
            ]
            leg1 = ax.legend(
                handles=svf_handles,
                loc='upper right',
                fontsize=14,
                frameon=True,
            )
            ax.add_artist(leg1)
            ax.legend(
                handles=test_handles,
                loc='center right',
                fontsize=14,
                frameon=True,
            )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f'Cross_Test_{name}_all_SVFs_peak_force_vs_cycle.png'
        if plotted:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  Saved per-filament cross-test plot: {out_path}")
        else:
            print(f"  No data to plot for filament {name}; skipping save.")
        plt.close()

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
        
        # Calculate peak force per specimen first, then aggregate by combination
        group_cols = ['Filament_Name', 'SVF_Percentage', 'Test_Type']
        if 'Folder_Name' in df.columns:
            group_cols.append('Folder_Name')
        peak_specimen = df.groupby(group_cols)['Force(Linear:Load) (kN)'].apply(lambda x: np.max(np.abs(x))).reset_index()
        # Now average across specimens within each Test_Type for each combination
        peak_data = peak_specimen.groupby(['Filament_Name', 'SVF_Percentage', 'Test_Type'])['Force(Linear:Load) (kN)'].mean().reset_index()
        peak_data = peak_data.rename(columns={'Force(Linear:Load) (kN)': 'Peak_Force_kN'})
        
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
        bars1 = ax1.bar(x_pos - width/2, normal_forces, width, label='Print 1', 
                         color='lightblue', alpha=0.8, edgecolor='darkblue')
        bars2 = ax1.bar(x_pos + width/2, reprint_forces, width, label='Print 2', 
                         color='lightcoral', alpha=0.8, edgecolor='darkred')
        
        # Add error bars representing standard deviation
        # ax1.errorbar(x_pos, mean_forces, yerr=[row['Std_Peak_Force_kN'] for _, row in stats_df.iterrows()], 
        #     fmt='none', color='black', capsize=5, capthick=2, zorder=3)
        # # Add mean peak force dots (no connecting line)
        # ax1.scatter(x_pos, mean_forces, color='black', s=30, zorder=4, label='Mean (± SD)')
        
        # Set up primary y-axis (force)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel('Filament-SVF Combinations', fontsize=14)
        ax1.set_ylabel('Peak Force (kN)', fontsize=14)
        # ax1.set_title('Test 1 (First Week) First Cycle Peak Force Analysis\nPeak Force by Combination with Individual RSD Calculations', 
        #              fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(combinations, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper left', fontsize=14)
        
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
                    color='black', linewidth=2, alpha=0.7, zorder=2, marker='o')

            # Add RSD value label
            ax2.annotate(f'{rsd:.1f}%', (x, rsd), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=14, color='black')
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylabel('Relative Standard Deviation (%)', fontsize=14, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Add custom legend for RSD visualization
        from matplotlib.lines import Line2D
        rsd_legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='RSD Chain (Normal ↔ Reprint)')
        ]
        # ax2.legend(handles=rsd_legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path('Test_1_First_Cycle_Peak_Force_Histogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Test 1 first cycle peak force histogram saved as: {output_path}")
        
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

def plot_cycle_peak_force_histogram(csv_path, cycle_num):
    """
    Create a histogram for a given cycle with mean dots (no connecting line) and error bars (± SD),
    plus RSD visualization on a secondary axis.
    """
    print(f"\nCreating cycle {cycle_num} peak force histogram...")
    try:
        df = pd.read_csv(csv_path)
        df_cycle = df[df['Cycle_Number'] == cycle_num] if 'Cycle_Number' in df.columns else df.copy()

        peak_data = df_cycle.groupby(['Filament_Name', 'SVF_Percentage', 'Test_Type'])['Force(Linear:Load) (kN)'].apply(
            lambda x: np.max(np.abs(x))
        ).reset_index()
        peak_data.columns = ['Filament_Name', 'SVF_Percentage', 'Test_Type', 'Peak_Force_kN']

        stats_data = []
        for (filament, svf), group in peak_data.groupby(['Filament_Name', 'SVF_Percentage']):
            normal_peak = group[group['Test_Type'] == 'normal']['Peak_Force_kN'].values
            reprint_peak = group[group['Test_Type'] == 'reprint']['Peak_Force_kN'].values
            all_peaks = []
            if len(normal_peak) > 0:
                all_peaks.append(normal_peak[0])
            if len(reprint_peak) > 0:
                all_peaks.append(reprint_peak[0])
            if len(all_peaks) >= 2:
                mean_peak = np.mean(all_peaks)
                std_peak = np.std(all_peaks, ddof=1)
                rsd = (std_peak / mean_peak) * 100 if mean_peak != 0 else 0
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
            print(f"No data found for cycle {cycle_num} histogram creation.")
            return None

        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        combinations = []
        mean_forces = []
        rsd_values = []
        normal_forces = []
        reprint_forces = []
        for _, row in stats_df.iterrows():
            combinations.append(f"{row['Filament_Name']}\nSVF {row['SVF_Percentage']}%")
            mean_forces.append(row['Mean_Peak_Force_kN'])
            rsd_values.append(row['RSD_Percent'])
            normal_forces.append(row['Normal_Peak'])
            reprint_forces.append(row['Reprint_Peak'])

        x_pos = np.arange(len(combinations))
        width = 0.35
        ax1.bar(x_pos - width/2, normal_forces, width, label='Print 1', 
                color='lightblue', alpha=0.8, edgecolor='darkblue')
        ax1.bar(x_pos + width/2, reprint_forces, width, label='Print 2', 
                color='lightcoral', alpha=0.8, edgecolor='darkred')
        # ax1.errorbar(x_pos, mean_forces, yerr=[row['Std_Peak_Force_kN'] for _, row in stats_df.iterrows()], 
        #               fmt='none', color='black', capsize=5, capthick=2, zorder=3)
        # ax1.scatter(x_pos, mean_forces, color='black', s=30, zorder=4, label='Mean (± SD)')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel('Filament-SVF Combinations', fontsize=14)
        ax1.set_ylabel('Peak Force (kN)', fontsize=14)
        # ax1.set_title(f'Test 1 Cycle {cycle_num} Peak Force Analysis\nPeak Force by Combination with Individual RSD Calculations', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(combinations, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper left', fontsize=14)

        ax2 = ax1.twinx()
        colors = plt.cm.Set3(np.linspace(0, 1, len(combinations)))
        for i, (x, rsd, normal, reprint) in enumerate(zip(x_pos, rsd_values, normal_forces, reprint_forces)):
            combo_color = colors[i]
            mean_val = (normal + reprint) / 2 if (normal is not None and reprint is not None) else 0
            if mean_val != 0:
                rsd_normal = ((normal - mean_val) / mean_val) * 100 + rsd
                rsd_reprint = ((reprint - mean_val) / mean_val) * 100 + rsd
                ax2.plot([x-0.1, x+0.1], [rsd_normal, rsd_reprint], color='black', linewidth=2, alpha=0.7, zorder=2, marker='o')
            ax2.annotate(f'{rsd:.1f}%', (x, rsd), textcoords="offset points", 
                         xytext=(0,15), ha='center', fontsize=14, color='black')
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylabel('Relative Standard Deviation (%)', fontsize=14, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        from matplotlib.lines import Line2D
        # ax2.legend(handles=[Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='RSD Chain (Normal ↔ Reprint)')], loc='upper right', fontsize=14)

        plt.tight_layout()
        output_path = Path(f'Test_1_Cycle_{cycle_num}_Peak_Force_Histogram.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Test 1 cycle {cycle_num} peak force histogram saved as: {output_path}")
        print(f"\nSummary Statistics:")
        print(f"Number of combinations analyzed: {len(stats_df)}")
        print(f"Mean peak force range: {stats_df['Mean_Peak_Force_kN'].min():.3f} - {stats_df['Mean_Peak_Force_kN'].max():.3f} kN")
        print(f"RSD range: {stats_df['RSD_Percent'].min():.1f}% - {stats_df['RSD_Percent'].max():.1f}%")
        print(f"Average RSD: {stats_df['RSD_Percent'].mean():.1f}%")
        return output_path
    except Exception as e:
        print(f"Error creating cycle {cycle_num} peak force histogram: {e}")
        return None

def compute_cycle_energy_loss(df, cycle_num):
    """
    Compute energy loss (hysteresis loop area) for a given cycle using the shoelace formula
    over the F-x closed path. Units: kN*mm = Joules.

    Returns a float (J) or None if insufficient data.
    """
    try:
        if df is None or 'Total Cycles' not in df.columns:
            return None
        sub = df[df['Total Cycles'] == cycle_num]
        if sub.empty or len(sub) < 3:
            return None
        # Coerce and drop NaNs
        x = pd.to_numeric(sub['Displacement(Linear:Digital Position) (mm)'], errors='coerce').to_numpy()
        y = pd.to_numeric(sub['Force(Linear:Load) (kN)'], errors='coerce').to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 3:
            return None
        # Close the loop for shoelace area computation
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        area = 0.5 * np.abs(np.sum(x_closed[:-1] * y_closed[1:] - x_closed[1:] * y_closed[:-1]))
        # kN*mm equals Joules
        return float(area)
    except Exception:
        return None

def _parse_filament_svf_from_name(folder_name):
    try:
        import re
        # Normalize separators
        name = str(folder_name).replace('_', ' ').strip()
        parts = name.split()

        candidates = []
        for i, token in enumerate(parts):
            # Skip the date-like first token (e.g., 20250905)
            if i == 0 and token.isdigit() and len(token) >= 6:
                continue
            # Pattern like '95-35'
            if '-' in token:
                t1, t2 = token.split('-', 1)
                if t1.isdigit() and t2.isdigit():
                    candidates.append((int(t1), int(t2)))
            # Pattern like '95 35'
            if token.isdigit() and i + 1 < len(parts) and parts[i + 1].isdigit():
                candidates.append((int(token), int(parts[i + 1])))

        def plausible(f, s):
            return (80 <= f <= 99) and (10 <= s <= 60)

        # Prefer plausible pairs; fall back to last candidate
        chosen = None
        for f, s in candidates:
            if plausible(f, s):
                chosen = (f, s)
        if chosen is None and candidates:
            chosen = candidates[-1]

        if not chosen:
            return None, None

        filament, svf = chosen
        if svf == 21:
            svf = 20
        return filament, svf
    except Exception:
        return None, None
        if svf == 21:
            svf = 20
        return filament, svf
    except Exception:
        return None, None

def plot_test1_average_energy_loss_histogram(base_path, cycles=[1, 100, 1000]):
    """
    Build a histogram (grouped bar chart) showing the average energy loss between the two samples
    (normal and reprint) of each (filament, svf) combination for the specified cycles in Test 1 (0 WK).

    Saves: 'Test_1_Average_Energy_Loss_Histogram.png' in the workspace root.
    """
    print("\nCreating Test 1 average energy loss histogram (cycles: {} )...".format(cycles))
    week_folder = '0 WK'
    week_path = Path(base_path) / week_folder
    if not week_path.exists():
        print(f"  Warning: {week_folder} folder not found; skipping energy loss histogram.")
        return None

    # Collect energy loss per (filament, svf, cycle) across samples
    # data[(filament, svf)][cycle] -> list of values from samples
    data = {}

    for folder in week_path.iterdir():
        if not folder.is_dir():
            continue
        filament, svf = _parse_filament_svf_from_name(folder.name)
        if filament is None or svf is None:
            continue
        tracking_file = find_tracking_csv(folder)
        if not tracking_file:
            continue
        df = load_tracking_data(tracking_file)
        if df is None:
            continue
        key = (filament, svf)
        if key not in data:
            data[key] = {c: [] for c in cycles}
        for c in cycles:
            val = compute_cycle_energy_loss(df, c)
            if val is not None:
                data[key][c].append(val)

    # Prepare plotting arrays
    if not data:
        print("  No energy loss data found for Test 1; skipping plot.")
        return None

    # Sort combinations for stable order
    combos = sorted(list(data.keys()))
    # Compute averages per cycle per combo
    avg_matrix = []  # rows per cycle in 'cycles' order; columns per combo
    for c in cycles:
        row = []
        for combo in combos:
            vals = data.get(combo, {}).get(c, [])
            row.append(float(np.mean(vals)) if vals else np.nan)
        avg_matrix.append(row)

    # Plot grouped bars per combo with one bar per cycle
    num_combos = len(combos)
    x = np.arange(num_combos)
    width = min(0.2, 0.8 / max(1, len(cycles)))
    fig_width = max(12, 0.6 * num_combos + 6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    bars = []
    for idx, c in enumerate(cycles):
        offsets = (idx - (len(cycles)-1)/2) * (width + 0.02)
        heights = np.array(avg_matrix[idx])
        b = ax.bar(x + offsets, heights, width, label=f'Cycle {c}', color=colors[idx % len(colors)], alpha=0.9)
        bars.append(b)

    # X labels as "TPUXXA SVF YY%"
    combo_labels = [f"{get_filament_name(f)} SVF {s}%" for (f, s) in combos]
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=45, ha='right')
    ax.set_xlabel('Filament–SVF Combinations', fontsize=14)
    ax.set_ylabel('Average Energy Loss (J)', fontsize=14)
    # ax.set_title('Test 1 Average Energy Loss per Combination\nCycles 1, 100, and 1000 (mean across samples) — 1 kN·mm = 1 J', fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = Path('Test_1_Average_Energy_Loss_Histogram.png')
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  Saved energy loss histogram: {out_path}")
    finally:
        plt.close()
    return out_path

def create_first_test_cycle_csv(base_path, cycle_num):
    """
    Extract specified cycle data from Test 1 (0 WK) tracking CSV files and create a CSV file.
    Returns the path to the created CSV or None if no data was found.
    """
    print(f"\nCreating Test 1 cycle {cycle_num} CSV file...")
    week_folder = '0 WK'
    test_name = 'Test 1'
    week_path = Path(base_path) / week_folder

    if not week_path.exists():
        print(f"  Warning: {week_folder} folder not found; cannot create cycle {cycle_num} CSV.")
        return None

    rows = []
    for folder in week_path.iterdir():
        if not folder.is_dir():
            continue
        filament, svf = extract_parameters_from_filename(folder.name)
        if filament is None or svf is None:
            # Try robust parser as fallback
            filament, svf = _parse_filament_svf_from_name(folder.name)
        if filament is None or svf is None:
            continue
        test_type = 'reprint' if 'REPRINT' in folder.name.upper() else 'normal'
        tracking_file = find_tracking_csv(folder)
        if not tracking_file:
            continue
        df = load_tracking_data(tracking_file)
        if df is None or 'Total Cycles' not in df.columns:
            continue
        sub = df[df['Total Cycles'] == cycle_num]
        if sub.empty:
            continue
        filament_name = get_filament_name(filament)
        for _, r in sub.iterrows():
            rows.append({
                'Filament_Name': filament_name,
                'SVF_Percentage': svf,
                'Test_Type': test_type,
                'Folder_Name': folder.name,
                'Cycle_Number': int(cycle_num),
                'Displacement(Linear:Digital Position) (mm)': float(r['Displacement(Linear:Digital Position) (mm)']),
                'Force(Linear:Load) (kN)': float(r['Force(Linear:Load) (kN)'])
            })

    if not rows:
        print(f"  No data found for cycle {cycle_num} in {week_folder}.")
        return None

    out_df = pd.DataFrame(rows)
    out_path = Path(f'Test_1_Cycle_{cycle_num}_Data.csv')
    try:
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote CSV: {out_path}")
        return out_path
    except Exception as e:
        print(f"  Failed to write CSV for cycle {cycle_num}: {e}")
        return None

def main():
    """
    Main analysis function
    """
    # Set the base path to the directory where this script is located
    base_path = Path(__file__).parent.resolve()
    
    print("Starting cyclic compression data analysis...")
    
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

        # NEW: Plot all average peak force vs cycle curves for all combinations on one graph
        print(f"Plot X: All combinations average peak force vs cycle - {test_name}...")
        plot_all_combinations_peak_force_vs_cycle(data_by_combination, test_name)

        print(f"Plot 2: Peak force vs cycle grouped by filament type (same SVF, different TPU types) - {test_name}...")
        plot_by_filament_type(data_by_combination, test_name)

        print(f"Plot 3: Peak force vs cycle grouped by SVF percentage (same TPU filament, different SVF) - {test_name}...")
        plot_by_svf_percentage(data_by_combination, test_name)

        # Generate VS groups plots for all tests
        print(f"Plot 4a: Peak force vs cycle comparison for similar vertical stiffnesses - {test_name}...")
        vs_groups_result = group_similar_vs(data_by_combination, tolerance=40)
        vs_groups, averaged_combinations = vs_groups_result
        if vs_groups:
            plot_vs_groups_peak_force_comparison(data_by_combination, vs_groups, averaged_combinations, test_name)
            print(f"Generated {len(vs_groups)} VS groups peak force comparison plots for {test_name}")
            print(f"Plot 4b: Force vs displacement by VS groups (similar vertical stiffnesses) - {test_name}...")
            plot_specific_cycles_by_vs_groups(data_by_combination, vs_groups, averaged_combinations, test_name, [1, 100, 1000])
            print(f"Generated {len(vs_groups)} VS groups force-displacement plots for {test_name}")
            # Also generate the requested pairwise comparisons with overlaid cycles on one figure
            requested_pairs = [
                ((87, 20), (90, 20)),
                ((87, 35), (95, 20)),
                ((90, 50), (95, 35)),
            ]
            print(f"Plot 4c: Pairwise force–displacement comparisons (overlaid cycles) - {test_name}...")
            plot_pairwise_force_displacement_overlaid(data_by_combination, test_name, requested_pairs, [1, 100, 1000])
            # And the pairwise peak force vs cycle plots (the ones you mentioned)
            print(f"Plot 4d: Pairwise peak force vs cycle - {test_name}...")
            plot_pairwise_peak_force_vs_cycle(data_by_combination, test_name, requested_pairs)
        else:
            print(f"No VS groups found for {test_name}")

        # Generate normal vs reprint comparison plots for Test 1 only
        if test_name == 'Test 1':
            print(f"Plot 5: Normal vs Reprint peak force comparison - {test_name}...")
            plot_normal_vs_reprint_comparison(data_by_combination, test_name)
            print(f"Plot 6: TPU95A 50% detailed comparison - {test_name}...")
            create_tpu95a_50_detailed_comparison(data_by_combination, test_name)
            print(f"Plot 7: Vertical stiffness at specific cycles (1st, 100th, 1000th) - {test_name}...")
            plot_vertical_stiffness_at_specific_cycles(data_by_combination, test_name, [1, 100, 1000])

        print(f"\n{test_name} analysis complete! Plots saved in: {output_base}")
        print(f"Generated plots in:")
        print(f"  - {test_name} vertical_stiffness_comparison/: Peak force vs cycle comparison plots for similar vertical stiffnesses")
        print(f"  - {test_name} force_displacement_cycles/: Force-displacement cycle plots")

    print(f"\n{'='*60}")
    print("All individual tests analysis complete!")
    print(f"{'='*60}")

    # Generate First Test First Cycle CSV and histogram plots
    print(f"\n{'='*60}")
    print("Creating First Test First Cycle CSV and histogram analysis...")
    print(f"{'='*60}")

    # Create the First Test First Cycle CSV
    csv_path = create_first_test_first_cycle_csv(base_path)

    if csv_path and csv_path.exists():
        print(f"\nGenerating histogram plots from: {csv_path}")
        # Generate peak force histogram
        plot_first_cycle_peak_force_histogram(csv_path)
        # Generate vertical stiffness histogram
        plot_first_cycle_vertical_stiffness_histogram(csv_path)
        # Also generate 100th and 1000th cycle peak force histograms
        csv100 = create_first_test_cycle_csv(base_path, 100)
        if csv100 and Path(csv100).exists():
            plot_cycle_peak_force_histogram(csv100, 100)
        else:
            print("Skipping 100th cycle histogram - CSV not created")

        csv1000 = create_first_test_cycle_csv(base_path, 1000)
        if csv1000 and Path(csv1000).exists():
            plot_cycle_peak_force_histogram(csv1000, 1000)
        else:
            print("Skipping 1000th cycle histogram - CSV not created")
        print("\nHistogram analysis complete!")
    else:
        print("\nSkipping histogram generation - CSV file not created")

    # Generate cross-test comparison plots
    print(f"\n{'='*60}")
    print("Generating cross-test comparison plots...")
    print(f"{'='*60}")

    # Plot peak force vs cycle for every filament-SVF combination across all tests
    plot_cross_test_peak_force_comparison(base_path)
    # Plot specific cycles for every filament-SVF combination across all tests
    plot_cross_test_cycle_comparison(base_path, [1, 100, 1000])
    # Plot per-filament (all SVFs, all 4 tests on same figure)
    plot_cross_test_peak_force_by_filament(base_path)

    # Create energy loss histogram for Test 1 (0 WK)
    print(f"\n{'='*60}")
    print("Generating Test 1 energy loss histogram (cycles 1, 100, 1000)...")
    print(f"{'='*60}")
    plot_test1_average_energy_loss_histogram(base_path, cycles=[1, 100, 1000])

    # Create VS comparison across cycles for Test 1
    print(f"\n{'='*60}")
    print("Generating Test 1 vertical stiffness comparison across cycles (1, 100, 1000)...")
    print(f"{'='*60}")
    plot_test1_vertical_stiffness_comparison_across_cycles(base_path, cycles=[1, 100, 1000])

# Place this function at module level, not inside main()
def plot_all_combinations_peak_force_vs_cycle(data_by_combination, test_name):
    """
    For a given week, plot all average peak force vs cycle curves for all filament-SVF combinations on one graph.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    print(f"\nPlotting all combinations' average peak force vs cycle for {test_name}...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 1) Group all series by (filament, svf), collecting both normal and reprint across VS buckets
    grouped = {}
    for _, combo_data in data_by_combination.items():
        filament = combo_data.get('filament')
        svf = combo_data.get('svf')
        if filament is None or svf is None:
            continue
        key = (filament, svf)
        if key not in grouped:
            grouped[key] = []  # list of (cycles, forces)

        # collect series from normal & reprint
        for set_name in ['normal', 'reprint']:
            if set_name in combo_data and combo_data[set_name]:
                set_data = combo_data[set_name]
                try:
                    if 'cycles' in set_data and 'peak_forces' in set_data and len(set_data['cycles']) and len(set_data['peak_forces']):
                        c_arr = np.array(set_data['cycles'])
                        f_arr = np.array(set_data['peak_forces'])
                    elif 'full_data' in set_data and set_data['full_data'] is not None:
                        c_arr, f_arr = find_peak_force_per_cycle(set_data['full_data'])
                    else:
                        c_arr, f_arr = np.array([]), np.array([])
                except Exception:
                    c_arr, f_arr = np.array([]), np.array([])
                if c_arr.size > 0 and f_arr.size > 0:
                    mask = c_arr != 1001
                    c_arr = c_arr[mask]
                    f_arr = f_arr[mask]
                    order = np.argsort(c_arr)
                    grouped[key].append((c_arr[order], f_arr[order]))

    # Helper to average a list of series into one
    def average_series(series_list):
        if not series_list:
            return None, None
        if len(series_list) == 1:
            return series_list[0]
        # try intersection of cycles
        common = None
        for c, _ in series_list:
            s = set(c.tolist())
            common = s if common is None else (common & s)
        if common:
            common = np.array(sorted(list(common)))
            aligned = []
            for c, f in series_list:
                idx_map = {int(cc): i for i, cc in enumerate(c)}
                vals = []
                for cc in common:
                    i = idx_map.get(int(cc))
                    if i is None:
                        vals = None
                        break
                    vals.append(f[i])
                if vals is not None:
                    aligned.append(np.array(vals))
            if aligned:
                return common, np.mean(aligned, axis=0)
        # fallback to truncation by min length
        min_len = min(len(c) for c, _ in series_list)
        if min_len == 0:
            return None, None
        cycles_ref = series_list[0][0][:min_len]
        avg_forces = np.mean([f[:min_len] for _, f in series_list], axis=0)
        return cycles_ref, avg_forces

    # 2) Prepare plotting order and colors; keep TPU95A 50% as one black line
    group_keys = sorted(grouped.keys())
    # Reserve color palette for all non-(95,50) groups
    non_special = [k for k in group_keys if not (k[0] == 95 and k[1] == 50)]
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(non_special))))
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(non_special)}

    plot_created = False
    # First plot non-special groups
    for key in non_special:
        series_list = grouped[key]
        cycles, forces = average_series(series_list)
        if cycles is None or forces is None or len(cycles) == 0:
            continue
        filament, svf = key
        filament_name = get_filament_name(filament)
        label = f"{filament_name} SVF {svf}%"
        ax.plot(cycles, forces, linewidth=2.0, color=color_map[key], label=label, alpha=0.95)
        plot_created = True

    # Then plot TPU95A 50% as aggregated black line if present
    if (95, 50) in grouped:
        cycles, forces = average_series(grouped[(95, 50)])
        if cycles is not None and forces is not None and len(cycles) > 0:
            ax.plot(cycles, forces, linewidth=2.6, color='black', label='TPU95A SVF 50% (Averaged)', alpha=1.0)
            plot_created = True

    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Peak Force (kN)', fontsize=12, fontweight='bold')
    ax.set_title(f'{test_name} - All Combinations\nAverage Peak Force vs Cycle', fontsize=14, fontweight='bold')
    if plot_created:
        n_items = len(group_keys)
        ncol = 1 if n_items <= 12 else 2 if n_items <= 24 else 3
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=ncol)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot in the vertical_stiffness_comparison folder
    output_dir = Path(f'{test_name} analysis_results/{test_name} vertical_stiffness_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{test_name}_all_combinations_peak_force_vs_cycle.png'
    if plot_created:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  All combinations peak force vs cycle plot saved as: {output_path}")
    else:
        print("  No valid series found to plot for all-combinations figure; skipping save.")
    plt.close()

if __name__ == "__main__":
    main()