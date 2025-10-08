"""Quasi-static compression data analysis

Generates:
    1. Individual engineering stress–strain curves for every test replicate (FINAL CYCLE ONLY when --final-only is used).
    2. Average engineering stress–strain curve (with ±1 SD band) for each filament type (87, 90, 95) using those final cycles.

Assumptions / Conventions:
  * Folder structure (example):
        Quasi-static/
            87-20 1/<nested test run folder>/<... (Peak-Valley).csv>
            87-20 2/...
            87-20 3/...
            90-20 1/...
            95-20 1/...
    Where the leading number before the first '-' is the filament hardness (87, 90, 95).
    The second number (e.g. 20) is kept but not averaged across (all appear identical here).
  * CSV format: Peak-Valley export with a preamble, then two header rows:
        "CycleCount","Axial Count ","Axial Displacement ","Axial Force "
        "CycleCount","cycles","mm","N"
    Followed by quoted numeric lines.
  * Engineering stress (MPa) = Force_N / A0_mm2.  (A0 = 60 mm * 60 mm = 3600 mm²)
  * Engineering strain (compressive) is taken as positive magnitude: strain = -Displacement_mm / L0_mm (since displacement is negative in compression).
    Also included: signed_strain (will be negative for compression) in the output CSV for completeness.
    * Force & displacement are zero-shifted by subtracting their first sample values so curves start nearer (0,0).
    * By default all cycles are processed; pass --final-only to restrict to the last cycle present in each file.
    * Use --y-cap-percentile (e.g. 99 or 97.5) to automatically cap y-axis (stress) at that positive magnitude percentile to avoid skew from spikes.
    * Curve clipping: by default (can disable with --no-stress-clip) if strain reaches 0.6, the stress value at 0.6 strain (interpolated) is S0. Any subsequent points where stress > stress_clip_multiplier * S0 (default 2.0) are removed (curve truncated at first violation).

CLI:
  python quasi_static_analysis.py --base "Quasi-static" --area 3600 --length 60

Outputs under: Quasi-static/analysis_results/
  individual_curves/filament_<filament>_test_<rep>.png
  individual_curves/filament_<filament>_test_<rep>.csv (clean stress-strain data)
  averages/filament_<filament>_average_curve.png
  averages/filament_<filament>_average_curve.csv (mean, std, n at each strain point)

If a filament has fewer than 2 valid replicates, an average curve is still produced (std = 0).

Dependencies: pandas, numpy, matplotlib, seaborn (already present in repository requirements).
"""

from __future__ import annotations

import re
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


DEFAULT_AREA_MM2 = 60 * 60  # 3600 mm^2
DEFAULT_GAUGE_LENGTH_MM = 60.0


def find_csv_file(test_dir: Path) -> Path | None:
    """Locate the Peak-Valley CSV inside an arbitrarily nested test directory.

    Preference: first file containing '(Peak-Valley).csv' (case-insensitive).
    Falls back to any .csv if none found.
    """
    candidates = list(test_dir.rglob('*Peak-Valley*.csv'))
    if not candidates:
        candidates = list(test_dir.rglob('*.csv'))
    if not candidates:
        return None
    # Pick shortest path (heuristic) then earliest alphabetically for determinism
    candidates.sort(key=lambda p: (len(p.parts), p.name))
    return candidates[0]


def parse_quasi_static_csv(path: Path) -> pd.DataFrame:
    """Parse the custom Peak-Valley CSV into a DataFrame with numeric columns.

    Returns columns: CycleCount (int), Displacement_mm (float), Force_N (float).
    """
    # Find the line number where the first header row appears
    header_line_idx = None
    with path.open('r', encoding='utf-8-sig', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('"CycleCount"'):
                header_line_idx = i
                break
    if header_line_idx is None:
        raise ValueError(f"Could not locate header row in file: {path}")

    # Read from header row onward without using header inference
    df_raw = pd.read_csv(path, header=None, skiprows=header_line_idx, encoding='utf-8-sig')
    # First row is textual column names, second row units; discard both
    if len(df_raw) < 3:
        raise ValueError(f"File too short to contain data rows: {path}")
    df = df_raw.iloc[2:].copy()
    df.columns = ['CycleCount', 'AxialCount', 'AxialDisplacement_mm', 'AxialForce_N']

    # Strip quotes & whitespace then convert
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('"', '').str.strip()
    df['CycleCount'] = pd.to_numeric(df['CycleCount'], errors='coerce').astype('Int64')
    df['AxialDisplacement_mm'] = pd.to_numeric(df['AxialDisplacement_mm'], errors='coerce')
    df['AxialForce_N'] = pd.to_numeric(df['AxialForce_N'], errors='coerce')
    df = df.dropna(subset=['CycleCount', 'AxialDisplacement_mm', 'AxialForce_N'])
    # Convert CycleCount to int (drop Int64 NA first)
    df = df[df['CycleCount'].notna()].copy()
    df['CycleCount'] = df['CycleCount'].astype(int)
    return df[['CycleCount', 'AxialDisplacement_mm', 'AxialForce_N']]


def compute_engineering_stress_strain(df: pd.DataFrame, area_mm2: float, gauge_length_mm: float) -> pd.DataFrame:
    """Append engineering stress (MPa) & strain columns.

    Displacement is negative in compression; we define:
      signed_strain = displacement_mm / gauge_length_mm  (negative for compression)
      strain = -signed_strain (positive magnitude)
      stress_MPa = Force_N / area_mm2 (will be negative for compression); provide also stress_MPa_mag = -stress (positive magnitude)
    Also zero-shift force & displacement relative to first sample so curves start near zero.
    """
    out = df.copy()
    if out.empty:
        return out
    # Zero shift
    out['Displacement_zero_mm'] = out['AxialDisplacement_mm'] - out['AxialDisplacement_mm'].iloc[0]
    out['Force_zero_N'] = out['AxialForce_N'] - out['AxialForce_N'].iloc[0]
    # Engineering strain (signed) from raw displacement (before zero so we preserve original magnitude context)
    out['signed_strain'] = out['AxialDisplacement_mm'] / gauge_length_mm
    out['strain'] = -out['signed_strain']  # positive compression
    out['stress_MPa'] = out['AxialForce_N'] / area_mm2
    out['stress_MPa_mag'] = -out['stress_MPa']  # positive compression
    # Also compute zero-shifted stress (for plotting alternative):
    out['stress_zero_MPa'] = out['Force_zero_N'] / area_mm2
    return out


FILE_PATTERN = re.compile(r'(?P<filament>87|90|95)[-_ ](?P<svf>20|35|50)\s+(?P<rep>\d+)', re.IGNORECASE)


def discover_tests(base_dir: Path) -> List[Dict]:
    """Traverse base_dir (e.g. 'Quasi-static') and collect test metadata.

    Returns list of dicts with keys: filament, svf, rep, root_dir, csv_path
    Only includes entries where a CSV is successfully located.
    """
    tests = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        match = FILE_PATTERN.search(child.name)
        if not match:
            # Skip folders that do not match pattern
            continue
        filament = int(match.group('filament'))
        svf = int(match.group('svf'))
        rep = int(match.group('rep'))
        csv_file = find_csv_file(child)
        if not csv_file:
            continue
        tests.append({
            'filament': filament,
            'svf': svf,
            'rep': rep,
            'root_dir': child,
            'csv_path': csv_file
        })
    return tests


def _clip_stress_curve(df: pd.DataFrame, strain_col: str, stress_col: str, trigger_strain: float, multiplier: float, debug: bool = False) -> pd.DataFrame:
    """Clip curve after it exceeds multiplier * stress_at(trigger_strain).

    - If maximum strain < trigger_strain, returns df unchanged.
    - Interpolates stress at trigger_strain using existing data (requires monotonic strain ascending).
    - Finds first index where stress > multiplier * stress_at_trigger and truncates (keeps data up to previous index).
    """
    if df.empty:
        return df
    if df[strain_col].max() < trigger_strain:
        return df
    # Prepare monotonic data (assume already sorted by processing, but enforce)
    work = df.sort_values(strain_col).drop_duplicates(subset=strain_col)
    try:
        stress_at_trigger = np.interp(trigger_strain, work[strain_col].values, work[stress_col].values)
    except Exception:
        return df
    if not np.isfinite(stress_at_trigger):
        return df
    threshold = multiplier * stress_at_trigger
    exceed = work[stress_col].values > threshold
    if not np.any(exceed):
        return df
    first_exceed_idx = np.argmax(exceed)
    # Keep all rows with strain <= strain at index before exceed (avoid including exceed sample)
    if first_exceed_idx == 0:
        # Everything exceeds; return empty to signal removal
        if debug:
            print(f"[DEBUG] Stress clipping removed entire curve (threshold={threshold:.3f})")
        return work.iloc[0:0]
    cutoff_strain = work.iloc[first_exceed_idx - 1][strain_col]
    clipped = df[df[strain_col] <= cutoff_strain].copy()
    if debug:
        print(f"[DEBUG] Stress clipping applied: trigger_strain={trigger_strain}, S0={stress_at_trigger:.3f}, threshold={threshold:.3f}, cutoff_strain={cutoff_strain:.4f}, kept_rows={len(clipped)} (from {len(df)})")
    return clipped


def build_individual_curves(tests: List[Dict], area_mm2: float, gauge_length_mm: float, final_only: bool = False, debug: bool = False, loading_only: bool = False,
                            stress_clip: bool = True, stress_clip_trigger: float = 0.6, stress_clip_multiplier: float = 2.0) -> List[Dict]:
    """Load and compute stress-strain for each test.

    Returns list of dicts with keys: filament, svf, rep, data (DataFrame).
    Filters out tests that fail to parse or produce empty data.
    """
    results = []
    for meta in tests:
        path = meta['csv_path']
        try:
            raw = parse_quasi_static_csv(path)
            if final_only and not raw.empty:
                all_cycles = raw['CycleCount'].unique()
                last_cycle = int(raw['CycleCount'].max())
                cycle_df = raw[raw['CycleCount'] == last_cycle]
                # Extra safeguard: ensure we didn't accidentally include earlier cycles
                if debug:
                    print(f"[DEBUG] {path.name}: cycles present={sorted(all_cycles)}, selecting final={last_cycle}, rows before={len(raw)}, final rows={len(cycle_df)}")
                raw = cycle_df.reset_index(drop=True)
                # Validation: assert only one unique cycle remains
                remaining_cycles = raw['CycleCount'].unique()
                if len(remaining_cycles) != 1 or remaining_cycles[0] != last_cycle:
                    if debug:
                        print(f"[DEBUG] Unexpected remaining cycles after filter: {remaining_cycles}")
                # If loading_only requested, truncate to loading branch (up to max compression magnitude)
                if loading_only and not raw.empty:
                    # Loading branch: from first sample until the most negative displacement (minimum value)
                    disp_col = 'AxialDisplacement_mm'
                    min_idx = raw[disp_col].idxmin()  # most negative displacement
                    loading_branch = raw.loc[:min_idx].copy()
                    if debug:
                        print(f"[DEBUG] {path.name}: loading_only applied, rows before={len(raw)}, after={len(loading_branch)}, min_idx={min_idx}")
                    raw = loading_branch.reset_index(drop=True)
            enriched = compute_engineering_stress_strain(raw, area_mm2, gauge_length_mm)
            # Apply stress clipping (using positive magnitude stress) after stress/strain computation
            if stress_clip and not enriched.empty:
                before_len = len(enriched)
                enriched = _clip_stress_curve(enriched, 'strain', 'stress_MPa_mag', stress_clip_trigger, stress_clip_multiplier, debug=debug)
                if debug and before_len != len(enriched):
                    print(f"[DEBUG] Clipped curve length {before_len} -> {len(enriched)} for {path.name}")
            if enriched.empty:
                continue
            meta_copy = dict(meta)
            meta_copy['data'] = enriched
            results.append(meta_copy)
        except Exception as e:
            print(f"[WARN] Failed to process {path}: {e}")
    return results


def _common_strain_axis(dfs: List[pd.DataFrame], strain_col: str = 'strain', n_points: int = 600) -> np.ndarray:
    """Compute a common strain axis limited to the minimum max strain across all dfs.
    """
    if not dfs:
        return np.array([])
    max_strains = [df[strain_col].max() for df in dfs if not df.empty]
    if not max_strains:
        return np.array([])
    common_max = min(max_strains)
    return np.linspace(0, common_max, n_points)


def average_filament_curves(individual: List[Dict]) -> Dict[int, Dict]:
    """Group data by filament and produce averaged stress-strain curves.

    Returns mapping:
      filament -> {
          'individual': [list of per-test dicts],
          'strain_axis': ndarray,
          'mean_stress_MPa': ndarray,
          'std_stress_MPa': ndarray,
          'n': int
      }
    Uses positive compression magnitude (stress_MPa_mag vs strain).
    """
    grouped: Dict[int, List[pd.DataFrame]] = {}
    meta_map: Dict[int, List[Dict]] = {}
    for test in individual:
        grouped.setdefault(test['filament'], []).append(test['data'])
        meta_map.setdefault(test['filament'], []).append(test)

    results: Dict[int, Dict] = {}
    for filament, dfs in grouped.items():
        strain_axis = _common_strain_axis(dfs)
        if strain_axis.size == 0:
            continue
        interpolated = []
        for df in dfs:
            # ensure monotonic strain for interpolation (data should progress with compression)
            df_sorted = df.sort_values('strain')
            # Drop duplicates in strain
            df_sorted = df_sorted.drop_duplicates(subset='strain')
            try:
                interp_stress = np.interp(strain_axis, df_sorted['strain'].values, df_sorted['stress_MPa_mag'].values)
                interpolated.append(interp_stress)
            except Exception:
                continue
        if not interpolated:
            continue
        arr = np.vstack(interpolated)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
        results[filament] = {
            'individual': meta_map[filament],
            'strain_axis': strain_axis,
            'mean_stress_MPa': mean,
            'std_stress_MPa': std,
            'n': arr.shape[0]
        }
    return results


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    root = base_dir / 'analysis_results'
    individual_dir = root / 'individual_curves'
    averages_dir = root / 'averages'
    for d in (individual_dir, averages_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {'root': root, 'individual': individual_dir, 'averages': averages_dir}


def _auto_cap(ax, y_values: np.ndarray, percentile: float | None):
    if percentile is None:
        return
    if y_values.size == 0:
        return
    cap = np.nanpercentile(y_values, percentile)
    if cap > 0:
        ax.set_ylim(0, cap * 1.02)


def plot_individual_curves(individual: List[Dict], dirs: Dict[str, Path], y_cap_percentile: float | None = None, final_only: bool = False, loading_only: bool = False):
    sns.set_context('talk')
    for test in individual:
        data = test['data']
        filament = test['filament']
        rep = test['rep']
        svf = test['svf']
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(data['strain'], data['stress_MPa_mag'], color='tab:blue', lw=2)
        ann_parts = []
        if final_only:
            ann_parts.append('Final Cycle')
        if loading_only:
            ann_parts.append('Loading Branch')
        if ann_parts:
            ax.text(0.02, 0.95, ', '.join(ann_parts), transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.6))
        _auto_cap(ax, data['stress_MPa_mag'].values, y_cap_percentile)
        ax.set_xlabel('Engineering Strain (Compression, +)')
        ax.set_ylabel('Engineering Stress (MPa, + Compression)')
        ax.set_title(f'Filament {filament} SVF {svf}% Rep {rep}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = dirs['individual'] / f'filament_{filament}_svf_{svf}_rep_{rep}.png'
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
        out_csv = dirs['individual'] / f'filament_{filament}_svf_{svf}_rep_{rep}.csv'
        # Export processed curve
        data[['CycleCount', 'AxialDisplacement_mm', 'AxialForce_N', 'strain', 'stress_MPa', 'stress_MPa_mag']].to_csv(out_csv, index=False)
        print(f"Saved individual curve: {out_png.name}")


def plot_average_curves(averages: Dict[int, Dict], dirs: Dict[str, Path], y_cap_percentile: float | None = None, final_only: bool = False, loading_only: bool = False):
    sns.set_context('talk')
    palette = {87: 'tab:green', 90: 'tab:orange', 95: 'tab:red'}
    for filament, info in sorted(averages.items()):
        fig, ax = plt.subplots(figsize=(7, 6))
        strain = info['strain_axis']
        mean = info['mean_stress_MPa']
        std = info['std_stress_MPa']
        n = info['n']
        color = palette.get(filament, 'tab:blue')
        ax.plot(strain, mean, color=color, lw=2.5, label=f'Filament {filament} (n={n})')
        if n > 1:
            ax.fill_between(strain, mean - std, mean + std, color=color, alpha=0.25, label='±1 SD')
        ax.set_xlabel('Engineering Strain (Compression, +)')
        ax.set_ylabel('Engineering Stress (MPa, + Compression)')
        title = f'Average Stress–Strain: Filament {filament}'
        suffix_parts = []
        if final_only:
            suffix_parts.append('Final Cycle')
        if loading_only:
            suffix_parts.append('Loading Branch')
        if suffix_parts:
            title += ' (' + ', '.join(suffix_parts) + ')'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        _auto_cap(ax, mean, y_cap_percentile)
        plt.tight_layout()
        out_png = dirs['averages'] / f'filament_{filament}_average_curve.png'
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
        # CSV export
        out_csv = dirs['averages'] / f'filament_{filament}_average_curve.csv'
        avg_df = pd.DataFrame({
            'strain': strain,
            'mean_stress_MPa': mean,
            'std_stress_MPa': std,
            'n': [n]*len(strain)
        })
        avg_df.to_csv(out_csv, index=False)
        print(f"Saved average curve: {out_png.name}")


def run_analysis(base: Path, area_mm2: float, gauge_length_mm: float, final_only: bool = False, y_cap_percentile: float | None = None,
                 debug: bool = False, loading_only: bool = False,
                 stress_clip: bool = True, stress_clip_trigger: float = 0.6, stress_clip_multiplier: float = 2.0):
    print(f"Scanning quasi-static tests under: {base}")
    tests_meta = discover_tests(base)
    if not tests_meta:
        print("No tests discovered. Check folder naming pattern like '87-20 1'.")
        return
    print(f" Discovered {len(tests_meta)} test folders.")
    individual_processed = build_individual_curves(
        tests_meta, area_mm2, gauge_length_mm,
        final_only=final_only, debug=debug, loading_only=loading_only,
        stress_clip=stress_clip, stress_clip_trigger=stress_clip_trigger, stress_clip_multiplier=stress_clip_multiplier
    )
    print(f" Processed {len(individual_processed)} tests successfully.")
    dirs = ensure_output_dirs(base)
    plot_individual_curves(individual_processed, dirs, y_cap_percentile=y_cap_percentile, final_only=final_only, loading_only=loading_only)
    averages = average_filament_curves(individual_processed)
    if not averages:
        print("No averages computed (insufficient data).")
        return
    plot_average_curves(averages, dirs, y_cap_percentile=y_cap_percentile, final_only=final_only, loading_only=loading_only)
    print(f"All outputs written to: {dirs['root']}\nDone.")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Quasi-static stress-strain analysis")
    p.add_argument('--base', type=str, default='Quasi-static', help='Base quasi-static data directory')
    p.add_argument('--area', type=float, default=DEFAULT_AREA_MM2, help='Cross-sectional area (mm^2)')
    p.add_argument('--length', type=float, default=DEFAULT_GAUGE_LENGTH_MM, help='Original gauge length (mm)')
    p.add_argument('--final-only', action='store_true', help='Use only the final (last) cycle from each test file.')
    p.add_argument('--y-cap-percentile', type=float, default=None, help='Percentile (e.g., 99, 97.5) to cap y-axis of stress plots (positive magnitude).')
    p.add_argument('--debug', action='store_true', help='Enable verbose debug output for cycle filtering.')
    p.add_argument('--loading-only', action='store_true', help='Keep only loading branch (start to max compression) within the final cycle.')
    p.add_argument('--no-stress-clip', action='store_true', help='Disable stress clipping beyond multiplier * stress at trigger strain (default trigger=0.6, multiplier=2).')
    p.add_argument('--stress-clip-trigger', type=float, default=0.6, help='Trigger strain at which reference stress S0 is taken (default 0.6).')
    p.add_argument('--stress-clip-multiplier', type=float, default=2.0, help='Multiplier of S0 defining clipping threshold (default 2.0).')
    return p


def main():
    args = build_arg_parser().parse_args()
    base = Path(args.base)
    if not base.exists():
        print(f"Base directory not found: {base}")
        return
    run_analysis(
        base,
        area_mm2=args.area,
        gauge_length_mm=args.length,
        final_only=args.final_only,
        y_cap_percentile=args.y_cap_percentile,
        debug=args.debug,
        loading_only=args.loading_only,
        stress_clip=not args.no_stress_clip,
        stress_clip_trigger=args.stress_clip_trigger,
        stress_clip_multiplier=args.stress_clip_multiplier
    )


if __name__ == '__main__':
    main()
