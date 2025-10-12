"""Quasi-static compression data analysis

Generates:
    1. Individual engineering stress–strain curves for every test replicate (FINAL CYCLE LOADING BRANCH by default).
    2. Average engineering stress–strain curve (with ±1 SD band) for each filament type (87, 90, 95) using those final-cycle loading branches.

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
    * By default, only the final cycle's LOADING branch is used for plots and averages (from start to max compression within that cycle).
    * Use --y-cap-percentile (e.g. 99 or 97.5) to automatically cap y-axis (stress) at that positive magnitude percentile to avoid skew from spikes.

CLI:
  python quasi_static_analysis.py --base "Quasi-static" --area 3600 --length 60

Outputs under: Quasi-static/analysis_results/
    individual_curves/filament_<filament>_test_<rep>.png
    individual_curves/filament_<filament>_test_<rep>.csv (clean stress–strain data for final-cycle loading branch)
    averages/filament_<filament>_average_curve.png
    averages/filament_<filament>_average_curve.csv (mean, std, n at each strain point; final-cycle loading branch only)

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
try:
    from brokenaxes import brokenaxes as BrokenAxes  # type: ignore
    HAS_BROKENAXES = True
except Exception:
    BrokenAxes = None  # type: ignore
    HAS_BROKENAXES = False


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


## Removed: stress clipping functionality (was truncating curves after a threshold). Keeping full curves now.


def build_individual_curves(tests: List[Dict], area_mm2: float, gauge_length_mm: float, final_only: bool = True, debug: bool = False, loading_only: bool = True) -> List[Dict]:
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


def _add_axes_data_margins(ax, x: float = 0.02, y: float = 0.05):
    """Add a small data margin so curves aren't flush against axes borders."""
    try:
        ax.margins(x=x, y=y)
    except Exception:
        pass


def _adjust_fig_margins(fig, left: float = 0.16, right: float = 0.98, bottom: float = 0.16, top: float = 0.98):
    """Adjust figure margins to add whitespace between content and image edges."""
    try:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    except Exception:
        pass


def _expand_interval(a: float, b: float, span: float, frac: float = 0.01) -> tuple[float, float]:
    """Expand [a,b] by a fraction of overall span to avoid edge clipping."""
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(span) or span <= 0:
        return a, b
    pad = frac * span
    return a - pad, b + pad


def plot_individual_curves(individual: List[Dict], dirs: Dict[str, Path], y_cap_percentile: float | None = None, final_only: bool = False, loading_only: bool = False):
    sns.set_context('talk')
    for test in individual:
        data = test['data']
        filament = test['filament']
        rep = test['rep']
        svf = test['svf']

        # Prepare data and title
        x = data['strain'].values
        y = data['stress_MPa_mag'].values
        title = f'Filament {filament} SVF {svf}% Rep {rep}'
        suffix = []
        if final_only:
            suffix.append('Final Cycle')
        if loading_only:
            suffix.append('Loading Branch')
        if suffix:
            title += ' (' + ', '.join(suffix) + ')'

        # Determine windows for broken y-axis: [0, 60%*peak] and [99%*peak, ~102%*peak]
        fig = None
        try:
            peak = float(np.nanmax(y)) if y.size else 0.0
        except Exception:
            peak = 0.0

        use_broken = HAS_BROKENAXES and np.isfinite(peak) and peak > 0
        if use_broken:
            # Y windows
            low_end = 0.055 * peak
            high_start = 0.95 * peak
            high_end = max(high_start * 1.001, 1.02 * peak)
            # Slightly expand y windows to avoid clipping markers/lines
            y_span = max(high_end, low_end) - 0.0 if np.isfinite(peak) else 1.0
            y0a, y0b = _expand_interval(0.0, low_end, y_span, 0.01)
            y1a, y1b = _expand_interval(high_start, high_end, y_span, 0.01)
            ylims = ((y0a, y0b), (y1a, y1b))
            # X windows derived from data where stress falls within Y windows
            x_arr = np.asarray(x); y_arr = np.asarray(y)
            xlims = None
            try:
                mask_low = (y_arr >= 0.0) & (y_arr <= low_end)
                mask_high = (y_arr >= high_start) & (y_arr <= high_end)
                if mask_low.sum() >= 2 and mask_high.sum() >= 2:
                    x_low_min = float(np.nanmin(x_arr[mask_low])); x_low_max = float(np.nanmax(x_arr[mask_low]))
                    x_high_min = float(np.nanmin(x_arr[mask_high])); x_high_max = float(np.nanmax(x_arr[mask_high]))
                    if x_low_max > x_low_min and x_high_max > x_high_min:
                        # Expand x windows slightly
                        x_span = float(np.nanmax(x_arr) - np.nanmin(x_arr)) if x_arr.size else 1.0
                        x0a, x0b = _expand_interval(x_low_min, x_low_max, x_span, 0.01)
                        x1a, x1b = _expand_interval(x_high_min, x_high_max, x_span, 0.01)
                        xlims = ((x0a, x0b), (x1a, x1b))
            except Exception:
                xlims = None
            fig = plt.figure(figsize=(7, 6))
            if xlims is not None:
                bax = BrokenAxes(ylims=ylims, xlims=xlims, hspace=0.05, wspace=0.05)
            else:
                bax = BrokenAxes(ylims=ylims, hspace=0.05)
            bax.plot(x, y, color='tab:blue', lw=2)
            # Manual axis labels using fig.text()
            fig.text(0.5, -0.02, 'Engineering Strain', ha='center', va='bottom', fontsize=18)
            fig.text(-0.02, 0.5, 'Engineering Stress', ha='left', va='center', rotation=90, fontsize=18)
            # bax.set_title(title)
            bax.tick_params(labelsize=14)
            bax.grid(True, alpha=0.3)
            # _adjust_fig_margins(fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(x, y, color='tab:blue', lw=2)
            _auto_cap(ax, y, y_cap_percentile)
            # Manual axis labels using fig.text()
            fig.text(0.5, 0.0, 'Engineering Strain', ha='center', va='bottom', fontsize=18)
            fig.text(0.0, 0.5, 'Engineering Stress', ha='left', va='center', rotation=90, fontsize=18)
            # ax.set_title(title)
            ax.grid(True, alpha=0.3)
            _add_axes_data_margins(ax)
            _adjust_fig_margins(fig)

        # plt.tight_layout()
        out_png = dirs['individual'] / f'filament_{filament}_svf_{svf}_rep_{rep}.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)
        out_csv = dirs['individual'] / f'filament_{filament}_svf_{svf}_rep_{rep}.csv'
        # Export processed curve
        data[['CycleCount', 'AxialDisplacement_mm', 'AxialForce_N', 'strain', 'stress_MPa', 'stress_MPa_mag']].to_csv(out_csv, index=False)
        print(f"Saved individual curve: {out_png.name}")


def plot_average_curves(averages: Dict[int, Dict], dirs: Dict[str, Path], y_cap_percentile: float | None = None, final_only: bool = False, loading_only: bool = False):
    sns.set_context('talk')
    palette = {87: 'tab:green', 90: 'tab:orange', 95: 'tab:red'}
    for filament, info in sorted(averages.items()):
        strain = info['strain_axis']
        mean = info['mean_stress_MPa']
        std = info['std_stress_MPa']
        n = info['n']
        color = palette.get(filament, 'tab:blue')
        title = f'Average Stress–Strain: Filament {filament}'
        suffix_parts = []
        if final_only:
            suffix_parts.append('Final Cycle')
        if loading_only:
            suffix_parts.append('Loading Branch')
        if suffix_parts:
            title += ' (' + ', '.join(suffix_parts) + ')'

        # Determine peak for window sizing
        try:
            peak = float(np.nanmax(mean + std)) if n > 1 else float(np.nanmax(mean))
        except Exception:
            peak = 0.0

        use_broken = HAS_BROKENAXES and np.isfinite(peak) and peak > 0
        if use_broken:
            # Y windows
            low_end = 0.055 * peak
            high_start = 0.95 * peak
            high_end = max(high_start * 1.001, 1.02 * peak)
            # Slightly expand y windows to avoid clipping
            y_span = max(high_end, low_end) - 0.0 if np.isfinite(peak) else 1.0
            y0a, y0b = _expand_interval(0.0, low_end, y_span, 0.01)
            y1a, y1b = _expand_interval(high_start, high_end, y_span, 0.01)
            ylims = ((y0a, y0b), (y1a, y1b))
            # X windows derived from mean stress windows
            s_arr = np.asarray(strain); m_arr = np.asarray(mean)
            xlims = None
            try:
                mask_low = (m_arr >= 0.0) & (m_arr <= low_end)
                mask_high = (m_arr >= high_start) & (m_arr <= high_end)
                if mask_low.sum() >= 2 and mask_high.sum() >= 2:
                    x_low_min = float(np.nanmin(s_arr[mask_low])); x_low_max = float(np.nanmax(s_arr[mask_low]))
                    x_high_min = float(np.nanmin(s_arr[mask_high])); x_high_max = float(np.nanmax(s_arr[mask_high]))
                    if x_low_max > x_low_min and x_high_max > x_high_min:
                        # Expand x windows slightly
                        x_span = float(np.nanmax(s_arr) - np.nanmin(s_arr)) if s_arr.size else 1.0
                        x0a, x0b = _expand_interval(x_low_min, x_low_max, x_span, 0.01)
                        x1a, x1b = _expand_interval(x_high_min, x_high_max, x_span, 0.01)
                        xlims = ((x0a, x0b), (x1a, x1b))
            except Exception:
                xlims = None
            fig = plt.figure(figsize=(7, 6))
            if xlims is not None:
                bax = BrokenAxes(ylims=ylims, xlims=xlims, hspace=0.05, wspace=0.05)
            else:
                bax = BrokenAxes(ylims=ylims, hspace=0.05)
            bax.plot(strain, mean, color=color, lw=2.5, label=f'Filament {filament} (n={n})')
            # if n > 1:
            #     bax.fill_between(strain, mean - std, mean + std, color=color, alpha=0.25, label='±1 SD')
            # Manual axis labels using fig.text()
            fig.text(0.5, -0.02, 'Engineering Strain', ha='center', va='bottom', fontsize=18)
            fig.text(-0.02, 0.5, 'Engineering Stress', ha='left', va='center', rotation=90, fontsize=18)
            bax.tick_params(labelsize=14)
            # bax.legend()
            bax.grid(True, alpha=0.3)
            # _adjust_fig_margins(fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(strain, mean, color=color, lw=2.5, label=f'Filament {filament} (n={n})')
            if n > 1:
                ax.fill_between(strain, mean - std, mean + std, color=color, alpha=0.25, label='±1 SD')
            # Manual axis labels using fig.text()
            fig.text(0.5, 0.0, 'Engineering Strain', ha='center', va='bottom', fontsize=18)
            fig.text(0.0, 0.5, 'Engineering Stress', ha='left', va='center', rotation=90, fontsize=18)
            # ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            _auto_cap(ax, mean, y_cap_percentile)
            _add_axes_data_margins(ax)
            _adjust_fig_margins(fig)

        # plt.tight_layout()
        out_png = dirs['averages'] / f'filament_{filament}_average_curve.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.25)
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


def run_analysis(base: Path, area_mm2: float, gauge_length_mm: float, final_only: bool = True, y_cap_percentile: float | None = None,
                 debug: bool = False, loading_only: bool = True):
    print(f"Scanning quasi-static tests under: {base}")
    tests_meta = discover_tests(base)
    if not tests_meta:
        print("No tests discovered. Check folder naming pattern like '87-20 1'.")
        return
    print(f" Discovered {len(tests_meta)} test folders.")
    individual_processed = build_individual_curves(
        tests_meta, area_mm2, gauge_length_mm,
        final_only=final_only, debug=debug, loading_only=loading_only,
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
    p.add_argument('--final-only', dest='final_only', action='store_true', help='Use only the final (last) cycle from each test file.')
    p.add_argument('--no-final-only', dest='final_only', action='store_false', help='Process all cycles (not recommended).')
    p.set_defaults(final_only=True)
    p.add_argument('--y-cap-percentile', type=float, default=None, help='Percentile (e.g., 99, 97.5) to cap y-axis of stress plots (positive magnitude).')
    p.add_argument('--debug', action='store_true', help='Enable verbose debug output for cycle filtering.')
    p.add_argument('--loading-only', dest='loading_only', action='store_true', help='Keep only loading branch (start to max compression) within the final cycle.')
    p.add_argument('--no-loading-only', dest='loading_only', action='store_false', help='Keep both loading and unloading branches.')
    p.set_defaults(loading_only=True)
    # Stress clipping removed; curves are no longer truncated.
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
    )


if __name__ == '__main__':
    main()
