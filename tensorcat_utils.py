"""
TensorCat Utilities
Helper functions for data loading and processing

Author: Sachin & Team TensorCat
Date: January 2026
"""

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional

from tensorcat_core import RiskConstants


def load_storm_data(filepath: str, max_events: Optional[int] = None) -> Tuple[Dict[str, Tensor], Optional[Tensor], Tensor, int]:
    """
    Load storm track data and calculate event occurrence rates
    
    Args:
        filepath: Path to parquet file with storm data
        max_events: Optional limit on number of events
        
    Returns:
        Tuple of (event_tracks, event_occurrence_rates, event_year_mapping, n_sim_years)
        - event_tracks: Dictionary with track tensors ready for pipeline
        - event_occurrence_rates: Tensor of annual occurrence rates per event (or None)
        - event_year_mapping: Tensor mapping event_id to year
        - n_sim_years: Total number of simulated years in catalog
    """
    print(f"\nLoading storm data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Standardize columns
    column_mapping = {
        'Maximum_wind_speed': 'vmax',
        'Radius_to_maximum_winds': 'rmax',
        'Longitude': 'lon',
        'Latitude': 'lat'
    }
    
    for old, new in column_mapping.items():
        if old in df.columns:
            df[new] = df[old]
    
    # Convert units
    if 'vmax' in df.columns and df['vmax'].max() > 200:
        df['vmax'] = df['vmax'] * RiskConstants.KTS_TO_MS
    
    if 'rmax' in df.columns:
        df['rmax'] = df['rmax'] * RiskConstants.NM_TO_KM
    
    # Get unique events
    if 'Event_ID' not in df.columns and 'TC_number' in df.columns:
        df['Event_ID'] = df['Year'].astype(str) + '_' + df['TC_number'].astype(str)
    
    unique_events = df['Event_ID'].unique()
    if max_events is not None:
        unique_events = unique_events[:max_events]
        
    # VECTORIZED - 1000x faster
    event_year_mapping = df.groupby('Event_ID', sort=False)['Year'].first().values

    # If you filtered to max_events
    if max_events is not None:
        event_year_mapping = event_year_mapping[:max_events]

    # Convert to tensor
    event_year_mapping = torch.tensor(event_year_mapping, dtype=torch.long)
    
    # Filter to selected events
    df = df[df['Event_ID'].isin(unique_events)].sort_values(['Event_ID'])
    
    n_events = len(unique_events)
    print(f"  Total events: {n_events:,}")
    
    # DETECT CATALOG TYPE AND CALCULATE RATES
    event_occurrence_rates = None
    if 'Year' in df.columns:
        n_catalog_years = df['Year'].nunique()
        year_min = df['Year'].min()
        year_max = df['Year'].max()
        n_sim_years = int(df['Year'].max() - df['Year'].min() + 1)
        
        print(f"\n  Catalog Year Information:")
        print(f"  Unique years in data: {n_catalog_years:,}")
        print(f"  Year range: {year_min:.0f} - {year_max:.0f}")
        print(f"  Total events loaded: {n_events:,}")
        print(f"  Total catalog years: {n_sim_years:,}")
        
        # Check if this is a stochastic catalog (many synthetic years)
        if n_catalog_years > 100:
            # Stochastic synthetic catalog
            annual_hurricane_rate = n_events / n_catalog_years
            individual_event_rate = 1.0 / n_catalog_years
            
            print(f"\n  Stochastic Synthetic Catalog Detected:")
            print(f"  Catalog years: {n_catalog_years:,}")
            print(f"  Annual hurricane rate: {annual_hurricane_rate:.2f} events/year")
            print(f"  Individual event rate: {individual_event_rate:.6f} (1/{n_catalog_years:,} per year)")
            
            # Equal probability for all events (stochastic catalog assumption)
            event_occurrence_rates = torch.ones(n_events, dtype=torch.float32) * individual_event_rate
            
            # Verify sum equals annual rate
            total_rate = event_occurrence_rates.sum().item()
            print(f"  Total annual frequency: {total_rate:.2f} events/year")
            print(f"  ✓ Probabilistic analysis enabled")
        else:
            # Historical catalog
            print(f"\n  Historical Catalog Detected:")
            print(f"  Catalog years: {n_catalog_years} (Year {year_min:.0f} - {year_max:.0f})")
            print(f"  Total events: {n_events}")
            print(f"  Each event represents one historical occurrence")
            print(f"  ⚠️  No annual rates assigned - use for historical analysis only")
    else:
        print(f"  ⚠️  No Year column found - cannot determine event rates")
        n_sim_years = 1000  # Default
    
    # Compute storm motion
    df['dx'] = df.groupby('Event_ID')['lon'].diff() * 111.32 * np.cos(np.radians(df['lat']))
    df['dy'] = df.groupby('Event_ID')['lat'].diff() * 111.32
    
    dt_hours = 6.0  # IBTrACS standard interval
    df['s_motion'] = (np.sqrt(df['dx']**2 + df['dy']**2) / dt_hours).fillna(0)
    df['s_motion'] = np.clip(df['s_motion'], 0, 150)  # Cap at realistic speeds
    df['heading'] = np.arctan2(df['dy'], df['dx']).fillna(0)
    
    # Compute Holland B parameter
    vmax_kts = df['vmax'] / RiskConstants.KTS_TO_MS
    df['B'] = np.clip(1.0 + (vmax_kts - 64) / 100, 1.0, 2.5)
    
    # Reshape into matrices
    counts = df.groupby('Event_ID', sort=False).size().values
    max_t = counts.max()
    n_e = len(unique_events)
    
    print(f"  Max timesteps: {max_t}")
    print(f"  Matrix shape: [{n_e:,} events × {max_t} timesteps]")
    
    def fast_reshape(col_name):
        arr = np.zeros((n_e, max_t), dtype=np.float32)
        raw = df[col_name].values
        offsets = np.zeros(len(counts) + 1, dtype=int)
        offsets[1:] = np.cumsum(counts)
        for i in range(n_e):
            length = counts[i]
            arr[i, :length] = raw[offsets[i]:offsets[i+1]]
            # Padded region stays ZERO
        return torch.from_numpy(arr.ravel())
    
    def create_validity_mask():
        """Create boolean mask indicating which timesteps are valid"""
        valid_mask = np.zeros((n_e, max_t), dtype=bool)
        for i in range(n_e):
            length = counts[i]
            valid_mask[i, :length] = True
        return torch.from_numpy(valid_mask.ravel())
    
    event_tracks = {
        'lon': fast_reshape('lon'),
        'lat': fast_reshape('lat'),
        'vmax': fast_reshape('vmax'),
        'rmax': fast_reshape('rmax'),
        'B': fast_reshape('B'),
        'speed': fast_reshape('s_motion'),
        'heading': fast_reshape('heading'),
        'valid_mask': create_validity_mask(),
        'n_events': n_e,
        'max_timesteps': max_t
    }

    return event_tracks, event_occurrence_rates, event_year_mapping, n_sim_years
