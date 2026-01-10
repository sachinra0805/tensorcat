"""
TensorCat Core Library
GPU-Native Catastrophe Modeling Four-Kernel Architecture

This module contains the core classes and functions for TensorCat.
Import these to use TensorCat in your own code.

Author: Sachin & Team TensorCat
Date: January 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from contextlib import nullcontext

# =============================================================================
# PHYSICAL CONSTANTS & CONFIGURATION
# =============================================================================

class RiskConstants:
    """Physical constants for catastrophe modeling"""
    # Wind thresholds (m/s)
    TROPICAL_STORM_THRESHOLD = 17.5  # 39 mph
    CAT1_THRESHOLD = 33.0  # 74 mph
    CAT2_THRESHOLD = 43.0  # 96 mph
    CAT3_THRESHOLD = 50.0  # 111 mph
    CAT4_THRESHOLD = 58.0  # 130 mph
    CAT5_THRESHOLD = 70.0  # 157 mph
    
    # Unit conversions
    KTS_TO_MS = 0.514444  # knots to m/s
    NM_TO_KM = 1.852  # nautical miles to km
    MS_TO_MPH = 2.23694  # m/s to mph
    MS_TO_KTS = 1.94384  # m/s to knots
    
    # Earth radius
    EARTH_RADIUS_KM = 6371.0
    
    # Physics parameters
    ASYMMETRY_COEFF = 0.55  # Storm asymmetry (Vickery & Wadhera 2008)
    ASYMMETRY_SIGMA = 0.5  # Radial decay of asymmetry
    INFLOW_ANGLE_RAD = np.radians(20.0)  # Surface wind inflow angle
    
    # Damage model parameters
    RESIDENTIAL_THRESHOLD_MS = 33.0  # ~74 mph
    COMMERCIAL_THRESHOLD_MS = 43.0  # ~96 mph

@dataclass
class KernelConfig:
    """Configuration for kernel execution"""
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32
    spatial_filter_radius_km: float = 400.0  # Maximum distance for event relevance
    temporal_batch_size: int = 50  # Events processed per temporal batch
    time_step_hours: float = 1.0  # Temporal resolution
    min_wind_threshold_ms: float = 17.5  # Minimum wind speed to consider
    fatigue_enabled: bool = True  # Enable progressive damage modeling
    debris_enabled: bool = True  # Enable debris accumulation
    use_mixed_precision: bool = False  # Use FP16 for 2x speedup
    spatial_batch_size: int = 500  # Batch size for spatial filtering
    enable_torch_compile: bool = False  # Use torch.compile for JIT optimization
    
# =============================================================================
# KERNEL 1: SPATIAL FILTERING
# =============================================================================

class SpatialFilteringKernel:
    """
    Kernel 1: Pre-filters which events affect which locations
    
    Purpose: Dramatically reduce computational burden by eliminating 
    event-location pairs that will never have meaningful impact
    
    Output: Sparse event-location mapping for downstream kernels
    """
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.device = config.device
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        
    def filter_events_by_location(
        self,
        event_tracks: Dict[str, Tensor],
        location_coords: Tensor,
        max_distance_km: Optional[float] = None
    ) -> Dict[str, Tensor]:
        """
        Filter events based on spatial proximity to locations
        
        Args:
            event_tracks: Dict with 'lon', 'lat' tensors [n_events, max_timesteps]
            location_coords: [n_locations, 2] tensor of [lon, lat]
            max_distance_km: Maximum distance for relevance (uses config if None)
            
        Returns:
            Dict with:
                - event_indices: [n_relevant] indices of relevant events
                - location_indices: [n_relevant] indices of affected locations
                - min_distances: [n_relevant] minimum distances (km)
                - relevance_matrix: Sparse [n_events, n_locations] boolean mask
        """
        max_dist = max_distance_km or self.config.spatial_filter_radius_km
        
        n_events = event_tracks['n_events']
        max_t = event_tracks['max_timesteps']
        n_locs = location_coords.shape[0]
        
        # Reshape track data
        track_lon = event_tracks['lon'].view(n_events, max_t)  # [E, T]
        track_lat = event_tracks['lat'].view(n_events, max_t)
        
        # Location coordinates
        loc_lon = location_coords[:, 0]  # [L]
        loc_lat = location_coords[:, 1]
        
        print(f"\n{'='*70}")
        print(f"KERNEL 1: SPATIAL FILTERING")
        print(f"{'='*70}")
        print(f"  Events: {n_events:,}")
        print(f"  Locations: {n_locs:,}")
        print(f"  Max distance: {max_dist:.1f} km")
        
        # For large location counts, process locations in chunks
        if n_locs > 200:
            print(f"  Large location count detected - using chunked processing...")
            return self._filter_events_chunked(
                event_tracks, location_coords, max_dist, n_events, max_t, n_locs,
                track_lon, track_lat
            )
        
        # Compute minimum distance for each event-location pair
        # [E, T, 1] - [1, 1, L] broadcasting
        start_time = time.time()
        
        min_distances = torch.full((n_events, n_locs), float('inf'), 
                                   device=self.device, dtype=self.config.dtype)
        
        # Process in batches to manage memory - OPTIMIZED for large scale
        event_batch_size = self.config.spatial_batch_size
        
        # Use autocast for mixed precision if enabled
        autocast_context = torch.amp.autocast('cuda') if self.use_amp else nullcontext()
        
        with autocast_context:
            for i in range(0, n_events, event_batch_size):
                batch_end = min(i + event_batch_size, n_events)
                
                batch_lon = track_lon[i:batch_end].unsqueeze(2)  # [B, T, 1]
                batch_lat = track_lat[i:batch_end].unsqueeze(2)
                
                # Compute distances for this batch
                distances = self._haversine_distance_batch(
                    batch_lon, batch_lat,
                    loc_lon.unsqueeze(0).unsqueeze(0),  # [1, 1, L]
                    loc_lat.unsqueeze(0).unsqueeze(0)
                )  # [B, T, L]
                
                # Get minimum over timesteps
                min_distances[i:batch_end] = distances.min(dim=1)[0]
                
                # Periodic GPU cache clearing for large datasets
                if (i // event_batch_size) % 20 == 0 and i > 0:
                    torch.cuda.empty_cache()
        
        # Create relevance mask
        relevance_matrix = min_distances < max_dist
        
        # Extract sparse indices
        event_indices, location_indices = torch.where(relevance_matrix)
        relevant_distances = min_distances[event_indices, location_indices]
        
        filter_time = time.time() - start_time
        
        n_pairs_total = n_events * n_locs
        n_pairs_filtered = len(event_indices)
        reduction_pct = 100.0 * (1.0 - n_pairs_filtered / n_pairs_total)
        
        print(f"\n  Filtering complete in {filter_time:.2f}s")
        print(f"  Total pairs: {n_pairs_total:,}")
        print(f"  Relevant pairs: {n_pairs_filtered:,}")
        print(f"  Reduction: {reduction_pct:.1f}%")
        print(f"  Memory saved: ~{(n_pairs_total - n_pairs_filtered) * 4 / 1e9:.2f} GB")
        
        return {
            'event_indices': event_indices,
            'location_indices': location_indices,
            'min_distances': relevant_distances,
            'relevance_matrix': relevance_matrix,
            'n_filtered_pairs': n_pairs_filtered
        }
    
    def _haversine_distance_batch(
        self, 
        lon1: Tensor, lat1: Tensor,
        lon2: Tensor, lat2: Tensor
    ) -> Tensor:
        """Vectorized haversine distance computation"""
        lon1_rad = torch.deg2rad(lon1)
        lat1_rad = torch.deg2rad(lat1)
        lon2_rad = torch.deg2rad(lon2)
        lat2_rad = torch.deg2rad(lat2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        return RiskConstants.EARTH_RADIUS_KM * c
    
    def _filter_events_chunked(
        self,
        event_tracks: Dict[str, Tensor],
        location_coords: Tensor,
        max_dist: float,
        n_events: int,
        max_t: int,
        n_locs: int,
        track_lon: Tensor,
        track_lat: Tensor
    ) -> Dict[str, Tensor]:
        """
        Chunked spatial filtering for large location counts
        
        Instead of computing [n_events × n_locs] matrix at once,
        process locations in chunks to avoid memory explosion
        """
        start_time = time.time()
        
        # Process locations in chunks of 100
        loc_chunk_size = 100
        
        # Storage for results across all chunks
        all_event_indices = []
        all_location_indices = []
        all_distances = []
        
        # Location coordinates
        loc_lon = location_coords[:, 0]
        loc_lat = location_coords[:, 1]
        
        autocast_context = torch.amp.autocast('cuda') if self.use_amp else nullcontext()
        
        # Process each location chunk
        for loc_start in range(0, n_locs, loc_chunk_size):
            loc_end = min(loc_start + loc_chunk_size, n_locs)
            chunk_size = loc_end - loc_start
            
            print(f"  Processing locations {loc_start:,}-{loc_end:,} ({100*loc_end/n_locs:.1f}%)")
            
            # Get location chunk
            chunk_lon = loc_lon[loc_start:loc_end]
            chunk_lat = loc_lat[loc_start:loc_end]
            
            # Compute distances for this location chunk
            min_distances_chunk = torch.full((n_events, chunk_size), float('inf'),
                                            device=self.device, dtype=self.config.dtype)
            
            with autocast_context:
                # Process events in batches
                event_batch_size = self.config.spatial_batch_size
                
                for i in range(0, n_events, event_batch_size):
                    batch_end = min(i + event_batch_size, n_events)
                    
                    batch_lon = track_lon[i:batch_end].unsqueeze(2)  # [B, T, 1]
                    batch_lat = track_lat[i:batch_end].unsqueeze(2)
                    
                    # Compute distances for this batch
                    distances = self._haversine_distance_batch(
                        batch_lon, batch_lat,
                        chunk_lon.unsqueeze(0).unsqueeze(0),  # [1, 1, C]
                        chunk_lat.unsqueeze(0).unsqueeze(0)
                    )  # [B, T, C]
                    
                    # Get minimum over timesteps
                    min_distances_chunk[i:batch_end] = distances.min(dim=1)[0]
            
            # Find relevant pairs in this chunk
            relevance_chunk = min_distances_chunk < max_dist
            event_idx_chunk, loc_idx_chunk = torch.where(relevance_chunk)
            
            # Adjust location indices to global coordinates
            loc_idx_chunk = loc_idx_chunk + loc_start
            
            # Get distances for relevant pairs
            distances_chunk = min_distances_chunk[event_idx_chunk, loc_idx_chunk - loc_start]
            
            # Store results
            all_event_indices.append(event_idx_chunk)
            all_location_indices.append(loc_idx_chunk)
            all_distances.append(distances_chunk)
            
            # Clear chunk memory
            del min_distances_chunk, relevance_chunk
            if (loc_start // loc_chunk_size) % 5 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        event_indices = torch.cat(all_event_indices)
        location_indices = torch.cat(all_location_indices)
        relevant_distances = torch.cat(all_distances)
        
        # Create sparse relevance matrix
        n_pairs_filtered = len(event_indices)
        
        filter_time = time.time() - start_time
        
        n_pairs_total = n_events * n_locs
        reduction_pct = 100.0 * (1.0 - n_pairs_filtered / n_pairs_total)
        
        print(f"\n  Filtering complete in {filter_time:.2f}s")
        print(f"  Total pairs: {n_pairs_total:,}")
        print(f"  Relevant pairs: {n_pairs_filtered:,}")
        print(f"  Reduction: {reduction_pct:.1f}%")
        print(f"  Memory saved: ~{(n_pairs_total - n_pairs_filtered) * 4 / 1e9:.2f} GB")
        
        return {
            'event_indices': event_indices,
            'location_indices': location_indices,
            'min_distances': relevant_distances,
            'relevance_matrix': None,  # Too large to materialize
            'n_filtered_pairs': n_pairs_filtered
        }

# =============================================================================
# KERNEL 2: TEMPORAL HAZARD STREAMING
# =============================================================================

class TemporalHazardStreamingKernel:
    """
    Kernel 2: Stream physics through time with progressive damage
    
    Purpose: Process temporal evolution of hazards WITHOUT materializing
    full spatiotemporal tensor. Captures:
    - Progressive wind speed increase/decrease
    - Storm asymmetry and translation effects
    - Wind-driven debris accumulation over time
    - Cascading infrastructure failures
    
    Key Innovation: Temporal streaming processes time slices in parallel
    across events without memory explosion
    """
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.device = config.device
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        
        # Adaptive batch sizing with auto-downgrade
        self.current_temporal_batch_size = config.temporal_batch_size
        self.original_batch_size = config.temporal_batch_size
        self.memory_safety_enabled = True
        self.downgrade_triggered = False
        
    def stream_temporal_hazard(
        self,
        event_tracks: Dict[str, Tensor],
        location_coords: Tensor,
        spatial_filter: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Stream temporal hazard evolution for filtered event-location pairs
        
        Args:
            event_tracks: Full track data [n_events, max_timesteps]
            location_coords: [n_locations, 2] location coordinates
            spatial_filter: Output from SpatialFilteringKernel
            
        Returns:
            Dict with:
                - peak_wind_speed: [n_filtered_pairs] maximum wind speed over time
                - exposure_duration: [n_filtered_pairs] hours above threshold
                - cumulative_wind_energy: [n_filtered_pairs] integrated wind energy
                - debris_load: [n_filtered_pairs] accumulated debris
                - timing_peak: [n_filtered_pairs] timestep of peak winds
        """
        print(f"\n{'='*70}")
        print(f"KERNEL 2: TEMPORAL HAZARD STREAMING")
        print(f"{'='*70}")
        
        n_events = event_tracks['n_events']
        max_t = event_tracks['max_timesteps']
        n_filtered = spatial_filter['n_filtered_pairs']
        
        # Ensure all tensors are on the correct device
        event_indices = spatial_filter['event_indices'].to(self.device)
        location_indices = spatial_filter['location_indices'].to(self.device)
        location_coords = location_coords.to(self.device)
        
        print(f"  Processing {n_filtered:,} filtered pairs")
        print(f"  Temporal resolution: {self.config.time_step_hours:.1f} hours")
        print(f"  Max timesteps: {max_t}")
        
        # Initialize output tensors
        peak_wind = torch.zeros(n_filtered, device=self.device, dtype=self.config.dtype)
        exposure_duration = torch.zeros(n_filtered, device=self.device, dtype=self.config.dtype)
        cumulative_energy = torch.zeros(n_filtered, device=self.device, dtype=self.config.dtype)
        debris_load = torch.zeros(n_filtered, device=self.device, dtype=self.config.dtype)
        timing_peak = torch.zeros(n_filtered, device=self.device, dtype=torch.long)
        
        # Reshape track data and move to device
        track_lon = event_tracks['lon'].view(n_events, max_t).to(self.device)
        track_lat = event_tracks['lat'].view(n_events, max_t).to(self.device)
        track_vmax = event_tracks['vmax'].view(n_events, max_t).to(self.device)
        track_rmax = event_tracks['rmax'].view(n_events, max_t).to(self.device)
        track_B = event_tracks['B'].view(n_events, max_t).to(self.device)
        track_speed = event_tracks['speed'].view(n_events, max_t).to(self.device)
        track_heading = event_tracks['heading'].view(n_events, max_t).to(self.device)
        track_valid_mask = event_tracks['valid_mask'].view(n_events, max_t).to(self.device)
        
        # Extract relevant location coordinates
        filtered_locs = location_coords[location_indices]  # [n_filtered, 2]
        
        start_time = time.time()
        
        # Adaptive batch sizing with memory monitoring
        batch_size = self.current_temporal_batch_size * 200
        
        # Check initial memory and adjust if needed
        if torch.cuda.is_available() and self.memory_safety_enabled:
            gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_reserved(0)) / 1e9
            estimated_batch_mem = (batch_size * max_t * 8) / 1e9  # 8 floats per pair-timestep
            
            # If estimated memory exceeds 80% of free memory, downgrade
            if estimated_batch_mem > gpu_mem_free * 0.8:
                safety_factor = 0.7 * gpu_mem_free / estimated_batch_mem
                old_batch = batch_size
                batch_size = int(batch_size * safety_factor)
                self.current_temporal_batch_size = batch_size // 200
                self.downgrade_triggered = True
                
                print(f"\n  ⚠️  AUTO-DOWNGRADE TRIGGERED:")
                print(f"     Original batch: {old_batch:,} pairs")
                print(f"     Reduced to: {batch_size:,} pairs")
                print(f"     Reason: Estimated memory {estimated_batch_mem:.2f}GB exceeds safe limit")
                print(f"     This will increase runtime but prevent OOM\n")
        
        # Use autocast for mixed precision if enabled
        autocast_context = torch.cuda.amp.autocast() if self.use_amp else nullcontext()
        
        oom_retry_count = 0
        max_retries = 2
        
        with autocast_context:
            for batch_start in range(0, n_filtered, batch_size):
                batch_end = min(batch_start + batch_size, n_filtered)
                batch_size_actual = batch_end - batch_start
                
                try:
                    # Extract batch event and location indices
                    batch_event_idx = event_indices[batch_start:batch_end]
                    batch_loc_idx = location_indices[batch_start:batch_end]
                    
                    # Get track data for batch events - already on correct device
                    batch_tracks = {
                        'lon': track_lon[batch_event_idx],  # [B, T]
                        'lat': track_lat[batch_event_idx],
                        'vmax': track_vmax[batch_event_idx],
                        'rmax': track_rmax[batch_event_idx],
                        'B': track_B[batch_event_idx],
                        'speed': track_speed[batch_event_idx],
                        'heading': track_heading[batch_event_idx],
                        'valid_mask': track_valid_mask[batch_event_idx]
                    }
                    
                    # Get locations for this batch - already on correct device
                    batch_locs = filtered_locs[batch_start:batch_end]  # [B, 2]
                    
                    # Stream through time for this batch
                    batch_results = self._stream_batch_temporal(
                        batch_tracks, batch_locs, max_t
                    )
                    
                    # Store results (convert back to float32 if using mixed precision)
                    peak_wind[batch_start:batch_end] = batch_results['peak_wind'].float()
                    exposure_duration[batch_start:batch_end] = batch_results['exposure_duration'].float()
                    cumulative_energy[batch_start:batch_end] = batch_results['cumulative_energy'].float()
                    debris_load[batch_start:batch_end] = batch_results['debris_load'].float()
                    timing_peak[batch_start:batch_end] = batch_results['timing_peak']
                    
                    # Reset OOM retry counter on success
                    oom_retry_count = 0
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and oom_retry_count < max_retries:
                        oom_retry_count += 1
                        
                        # Emergency downgrade - cut batch size in half
                        torch.cuda.empty_cache()
                        old_batch = batch_size
                        batch_size = batch_size // 2
                        self.current_temporal_batch_size = batch_size // 200
                        
                        print(f"\n  🚨 OOM DETECTED - Emergency downgrade #{oom_retry_count}")
                        print(f"     Batch size: {old_batch:,} → {batch_size:,} pairs")
                        print(f"     Retrying from batch {batch_start:,}...\n")
                        
                        # Retry this batch with smaller size
                        continue
                    else:
                        # If we've exhausted retries or it's a different error, raise
                        raise e
                
                # Progress reporting
                if (batch_start // batch_size) % 5 == 0:
                    progress = 100.0 * batch_end / n_filtered
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    eta = (n_filtered - batch_end) / rate if rate > 0 else 0
                    
                    # GPU memory monitoring
                    mem_str = ""
                    if torch.cuda.is_available():
                        gpu_mem_used = torch.cuda.memory_allocated(0) / 1e9
                        gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory - 
                                       torch.cuda.memory_reserved(0)) / 1e9
                        mem_str = f" | GPU: {gpu_mem_used:.2f}GB used, {gpu_mem_free:.2f}GB free"
                    
                    print(f"  Progress: {progress:.1f}% ({batch_end:,}/{n_filtered:,} pairs) | {rate:.0f} pairs/s | ETA: {eta:.1f}s{mem_str}")
                
                # Periodic GPU cache clearing for large datasets
                if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
                    torch.cuda.empty_cache()
        
        stream_time = time.time() - start_time
        
        print(f"\n  Temporal streaming complete in {stream_time:.2f}s")
        print(f"  Throughput: {n_filtered / stream_time:.0f} pairs/second")
        print(f"  Peak wind range: {peak_wind.min():.1f} - {peak_wind.max():.1f} m/s")
        print(f"  Mean exposure: {exposure_duration.mean():.1f} hours")
        
        if self.downgrade_triggered:
            reduction_pct = 100 * (1 - self.current_temporal_batch_size / self.original_batch_size)
            print(f"\n  ⚠️  Batch size auto-reduced by {reduction_pct:.0f}% to prevent OOM")
            print(f"     Original: {self.original_batch_size * 200:,} pairs/batch")
            print(f"     Final: {self.current_temporal_batch_size * 200:,} pairs/batch")
        
        return {
            'peak_wind_speed': peak_wind,
            'exposure_duration': exposure_duration,
            'cumulative_wind_energy': cumulative_energy,
            'debris_load': debris_load,
            'timing_peak': timing_peak,
            'event_indices': event_indices,
            'location_indices': location_indices
        }
    
    def _stream_batch_temporal(
        self,
        batch_tracks: Dict[str, Tensor],
        batch_locs: Tensor,
        max_t: int
    ) -> Dict[str, Tensor]:
        """
        Stream temporal evolution with VALIDITY MASK CHECK
        
        CRITICAL FIX: Skip invalid timesteps to prevent damage overestimation
        """
        batch_size = batch_tracks['lon'].shape[0]
        
        # Initialize accumulators
        peak_wind = torch.zeros(batch_size, device=self.device, dtype=self.config.dtype)
        exposure_duration = torch.zeros(batch_size, device=self.device, dtype=self.config.dtype)
        cumulative_energy = torch.zeros(batch_size, device=self.device, dtype=self.config.dtype)
        debris_load = torch.zeros(batch_size, device=self.device, dtype=self.config.dtype)
        timing_peak = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        
        # Previous wind speed for debris accumulation
        prev_wind = torch.zeros(batch_size, device=self.device, dtype=self.config.dtype)
        
        # Extract validity mask for this batch
        valid_mask = batch_tracks['valid_mask']  # [B, T]
        
        # Stream through timesteps
        for t in range(max_t):
            # Check validity mask first
            timestep_valid = valid_mask[:, t]  # [B] - boolean tensor
            
            # Skip computation if no valid events at this timestep
            if not timestep_valid.any():
                continue
            
            # Extract timestep data
            storm_lon = batch_tracks['lon'][:, t]  # [B]
            storm_lat = batch_tracks['lat'][:, t]
            vmax = batch_tracks['vmax'][:, t]
            rmax = batch_tracks['rmax'][:, t]
            B = batch_tracks['B'][:, t]
            speed = batch_tracks['speed'][:, t]
            heading = batch_tracks['heading'][:, t]
            
            # Compute distances from storm centers to locations
            distances = self._haversine_distance_simple(
                storm_lon, storm_lat,
                batch_locs[:, 0], batch_locs[:, 1]
            )  # [B]
            
            # Compute wind speed with full physics
            wind_speed = self._compute_wind_with_physics(
                distances, vmax, rmax, B, speed, heading,
                storm_lon, storm_lat, batch_locs
            )  # [B]
            
            # Apply mask: Zero out wind for invalid timesteps
            wind_speed = wind_speed * timestep_valid.float()
            
            # Update peak wind and timing (only for valid timesteps)
            is_new_peak = (wind_speed > peak_wind) & timestep_valid
            peak_wind = torch.where(is_new_peak, wind_speed, peak_wind)
            timing_peak = torch.where(is_new_peak, torch.full_like(timing_peak, t), timing_peak)
            
            # Update exposure duration (only count valid timesteps)
            above_threshold = (wind_speed > self.config.min_wind_threshold_ms) & timestep_valid
            exposure_duration += above_threshold.float() * self.config.time_step_hours
            
            # Update cumulative wind energy (only for valid timesteps)
            cumulative_energy += (wind_speed ** 2) * self.config.time_step_hours * timestep_valid.float()
            
            # Update debris load (only for valid timesteps)
            if self.config.debris_enabled:
                # Debris increases with high winds and wind changes
                wind_change = torch.abs(wind_speed - prev_wind)
                debris_increment = (wind_speed ** 1.5) * 0.01 + wind_change * 0.1
                debris_load += debris_increment * above_threshold.float()
                # Only update prev_wind for valid timesteps
                prev_wind = torch.where(timestep_valid, wind_speed, prev_wind)
        
        return {
            'peak_wind': peak_wind,
            'exposure_duration': exposure_duration,
            'cumulative_energy': cumulative_energy,
            'debris_load': debris_load,
            'timing_peak': timing_peak
        }
    
    def _compute_wind_with_physics(
        self,
        distances: Tensor,
        vmax: Tensor,
        rmax: Tensor,
        B: Tensor,
        speed: Tensor,
        heading: Tensor,
        storm_lon: Tensor,
        storm_lat: Tensor,
        loc_coords: Tensor
    ) -> Tensor:
        """
        Compute wind speed with full physics:
        - Holland (1980) wind profile
        - Storm asymmetry from translation
        - Inflow angle
        """
        # Holland profile (base symmetric wind)
        R_safe = torch.clamp(distances, min=0.1)
        ratio = rmax / R_safe
        W_symmetric = vmax * torch.sqrt((ratio ** B) * torch.exp(1.0 - ratio ** B))
        W_symmetric = torch.where(distances < 0.1, torch.zeros_like(W_symmetric), W_symmetric)
        
        # Storm asymmetry from translation (faster winds in forward quadrant)
        # Compute bearing from storm to location
        dlon = loc_coords[:, 0] - storm_lon
        dlat = loc_coords[:, 1] - storm_lat
        bearing_to_loc = torch.atan2(dlat, dlon)
        
        # Angle difference between storm heading and bearing to location
        angle_diff = bearing_to_loc - heading
        
        # Asymmetry factor (winds stronger ahead of storm motion)
        # Convert speed from km/h to m/s for asymmetry
        speed_ms = speed / 3.6
        asymmetry_decay = torch.exp(-(distances / (rmax * RiskConstants.ASYMMETRY_SIGMA)))
        asymmetry_factor = 1.0 + RiskConstants.ASYMMETRY_COEFF * speed_ms * torch.cos(angle_diff) * asymmetry_decay / vmax.clamp(min=1.0)
        
        # Apply asymmetry
        W_asymmetric = W_symmetric * asymmetry_factor.clamp(min=0.5, max=1.5)
        
        # Apply inflow angle (surface winds spiral inward)
        # This reduces wind speed slightly (typically 10-20%)
        inflow_reduction = torch.cos(torch.tensor(RiskConstants.INFLOW_ANGLE_RAD, device=self.device))
        W_final = W_asymmetric * inflow_reduction
        
        return W_final.clamp(min=0.0)
    
    def _haversine_distance_simple(
        self,
        lon1: Tensor, lat1: Tensor,
        lon2: Tensor, lat2: Tensor
    ) -> Tensor:
        """Simple haversine for 1D tensors"""
        lon1_rad = torch.deg2rad(lon1)
        lat1_rad = torch.deg2rad(lat1)
        lon2_rad = torch.deg2rad(lon2)
        lat2_rad = torch.deg2rad(lat2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        return RiskConstants.EARTH_RADIUS_KM * c

# =============================================================================
# KERNEL 3: VULNERABILITY ASSESSMENT
# =============================================================================

class VulnerabilityAssessmentKernel:
    """
    Kernel 3: PRODUCTION-GRADE Component-Level Vulnerability Assessment
    
    MAJOR UPGRADE from simple power functions to:
    - Component-level damage (roof, walls, windows, foundation)
    - Building characteristics (age, construction type, stories, mitigation)
    - Storm surge vulnerability
    - HAZUS empirical damage curves
    - Temporal fatigue modeling
    
    This is what industry cat models actually use.
    """
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.device = config.device
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        
        # HAZUS-based component vulnerability parameters
        # Format: {building_type: {component: {construction_quality: params}}}
        self.component_params = self._initialize_component_params()
        
        # Pre-compute lookup tables for fast damage evaluation
        self.vuln_lookup = self._precompute_vulnerability_tables()
        
    def _initialize_component_params(self) -> Dict:
        """Initialize HAZUS-based component vulnerability parameters"""
        
        params = {
            'residential': {
                'roof': {
                    # Roof is most vulnerable - fails first
                    'wood_frame': {'v50': 130.0, 'beta': 4.5, 'weight': 0.40},  # 40% of total damage
                    'masonry': {'v50': 150.0, 'beta': 4.3, 'weight': 0.40},
                    'engineered': {'v50': 165.0, 'beta': 4.0, 'weight': 0.40}
                },
                'walls': {
                    # Walls fail at higher winds
                    'wood_frame': {'v50': 115.0, 'beta': 2.2, 'weight': 0.30},
                    'masonry': {'v50': 130.0, 'beta': 2.0, 'weight': 0.30},
                    'engineered': {'v50': 145.0, 'beta': 1.8, 'weight': 0.30}
                },
                'windows': {
                    # Windows fail early but contribute less to total loss
                    'standard': {'v50': 85.0, 'beta': 3.0, 'weight': 0.20},
                    'impact_resistant': {'v50': 125.0, 'beta': 2.5, 'weight': 0.20},
                    'shuttered': {'v50': 150.0, 'beta': 2.0, 'weight': 0.20}
                },
                'foundation': {
                    # Foundation rarely fails but when it does, total loss
                    'slab': {'v50': 140.0, 'beta': 1.5, 'weight': 0.10},
                    'pier': {'v50': 120.0, 'beta': 1.8, 'weight': 0.10},
                    'basement': {'v50': 160.0, 'beta': 1.3, 'weight': 0.10}
                }
            },
            'commercial': {
                # Commercial buildings are generally stronger
                'roof': {
                    'metal': {'v50': 110.0, 'beta': 2.3, 'weight': 0.35},
                    'concrete': {'v50': 130.0, 'beta': 2.0, 'weight': 0.35}
                },
                'walls': {
                    'metal': {'v50': 125.0, 'beta': 2.0, 'weight': 0.35},
                    'concrete': {'v50': 145.0, 'beta': 1.8, 'weight': 0.35}
                },
                'windows': {
                    'standard': {'v50': 90.0, 'beta': 2.8, 'weight': 0.20},
                    'impact_resistant': {'v50': 130.0, 'beta': 2.3, 'weight': 0.20}
                },
                'foundation': {
                    'slab': {'v50': 150.0, 'beta': 1.4, 'weight': 0.10}
                }
            }
        }
        
        return params
    
    def _precompute_vulnerability_tables(self) -> Dict[str, Tensor]:
        """Pre-compute damage curves for all wind speeds (0-200 knots)"""
        wind_speeds = torch.arange(0, 201, 1, dtype=torch.float32, device=self.device)
        tables = {}
        
        for btype in ['residential', 'commercial']:
            for component in self.component_params[btype].keys():
                for quality in self.component_params[btype][component].keys():
                    params = self.component_params[btype][component][quality]
                    v50 = params['v50']
                    beta = params['beta']
                    
                    key = f"{btype}_{component}_{quality}"
                    
                    # Sigmoid vulnerability curve
                    x = (wind_speeds - v50) / v50
                    damage = 1.0 / (1.0 + torch.exp(-beta * x))
                    
                    tables[key] = damage
        
        return tables
    
    def assess_vulnerability(
        self,
        hazard_results: Dict[str, Tensor],
        building_type: str = 'residential',
        building_attributes: Optional[Dict[str, Tensor]] = None,
        construction_quality: str = 'wood_frame',
        window_protection: str = 'standard',
        foundation_type: str = 'slab',
        surge_depth: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        ENHANCED: Component-level vulnerability assessment
        
        Args:
            hazard_results: Output from TemporalHazardStreamingKernel
            building_type: 'residential' or 'commercial'
            building_attributes: Optional dict with:
                - 'age_years': [n_locations] building ages
                - 'stories': [n_locations] number of stories
                - 'year_built': [n_locations] construction year
            construction_quality: 'wood_frame', 'masonry', 'engineered' (residential)
                                  or 'metal', 'concrete' (commercial)
            window_protection: 'standard', 'impact_resistant', 'shuttered'
            foundation_type: 'slab', 'pier', 'basement'
            surge_depth: Optional [n_filtered_pairs] storm surge depth (meters)
            
        Returns:
            Dict with component-level and total damage ratios
        """
        print(f"\n{'='*70}")
        print(f"KERNEL 3: ENHANCED VULNERABILITY ASSESSMENT")
        print(f"{'='*70}")
        
        n_filtered = len(hazard_results['peak_wind_speed'])
        
        print(f"  Building type: {building_type}")
        print(f"  Construction: {construction_quality}")
        print(f"  Window protection: {window_protection}")
        print(f"  Foundation: {foundation_type}")
        print(f"  Component-level modeling: ENABLED")
        print(f"  Fatigue modeling: {'Enabled' if self.config.fatigue_enabled else 'Disabled'}")
        
        start_time = time.time()
        
        # Extract hazard metrics
        peak_wind = hazard_results['peak_wind_speed']
        exposure_duration = hazard_results['exposure_duration']
        duration_factor = torch.clamp(exposure_duration / 5.0, min=0.6, max=1.25)
        cumulative_energy = hazard_results['cumulative_wind_energy']
        debris_load = hazard_results['debris_load']
        
        # Convert to knots for vulnerability curves
        peak_wind_kt = peak_wind * RiskConstants.MS_TO_KTS
        
        # =====================================================================
        # COMPONENT-LEVEL DAMAGE ASSESSMENT
        # =====================================================================
        
        components = self.component_params[building_type]
        
        # 1. ROOF DAMAGE
        roof_params = components['roof'][construction_quality]
        roof_damage = self._compute_component_damage(
            peak_wind_kt, roof_params['v50'], roof_params['beta']
        )
        
        # 2. WALL DAMAGE
        wall_params = components['walls'][construction_quality]
        wall_damage = self._compute_component_damage(
            peak_wind_kt, wall_params['v50'], wall_params['beta']
        )
        
        # 3. WINDOW DAMAGE
        window_params = components['windows'][window_protection]
        window_damage = self._compute_component_damage(
            peak_wind_kt, window_params['v50'], window_params['beta']
        )
        
        # 4. FOUNDATION DAMAGE
        foundation_params = components['foundation'][foundation_type]
        foundation_damage = self._compute_component_damage(
            peak_wind_kt, foundation_params['v50'], foundation_params['beta']
        )
        
        # =====================================================================
        # WEIGHTED COMPONENT AGGREGATION
        # =====================================================================
        
        base_wind_damage = (
            roof_damage * roof_params['weight'] +
            wall_damage * wall_params['weight'] +
            window_damage * window_params['weight'] +
            foundation_damage * foundation_params['weight']
        )
        
        # =====================================================================
        # FATIGUE ADJUSTMENT (Progressive damage from sustained winds)
        # =====================================================================
        
        wind_damage = base_wind_damage * duration_factor
        if self.config.fatigue_enabled:
            # Fatigue increases damage for sustained high winds
            fatigue_factor = 1.0 + 0.08 * torch.tanh(cumulative_energy / 10000.0)
            wind_damage = wind_damage * fatigue_factor
        
        # =====================================================================
        # DEBRIS DAMAGE (Additional damage from wind-borne debris)
        # =====================================================================
        
        debris_damage = torch.zeros_like(base_wind_damage)
        if self.config.debris_enabled:
            debris_normalized = debris_load / (debris_load.max() + 1e-6)
            # Debris primarily damages windows and roof
            debris_damage = 0.12 * debris_normalized * (base_wind_damage ** 0.5)
        
        # =====================================================================
        # STORM SURGE DAMAGE (If surge data provided)
        # =====================================================================
        
        surge_damage = torch.zeros_like(base_wind_damage)
        if surge_depth is not None:
            # Surge damage increases with depth
            # 0m = 0%, 1m = 20%, 2m = 40%, 3m+ = 60%+
            surge_damage = torch.tanh(surge_depth / 2.0) * 0.6
            print(f"  Storm surge modeling: ENABLED")
            print(f"  Max surge depth: {surge_depth.max():.2f}m")
        
        # =====================================================================
        # BUILDING AGE DETERIORATION
        # =====================================================================
        
        age_factor = torch.ones_like(base_wind_damage)
        if building_attributes is not None and 'age_years' in building_attributes:
            ages = building_attributes['age_years'][hazard_results['location_indices']]
            # 1% increase in vulnerability per 10 years, max 40%
            age_factor = 1.0 + torch.clamp(ages / 1000.0, 0.0, 0.4)
        
        # =====================================================================
        # TOTAL DAMAGE AGGREGATION
        # =====================================================================
        
        # Wind, debris, and surge damage are ADDITIVE but capped at 1.0
        total_damage = (wind_damage + debris_damage + surge_damage) * age_factor
        total_damage = total_damage.clamp(max=1.0)
        
        assess_time = time.time() - start_time
        
        print(f"\n  Vulnerability assessment complete in {assess_time:.2f}s")
        print(f"  Component damage breakdown (mean):")
        print(f"    Roof: {roof_damage.mean():.3f}")
        print(f"    Walls: {wall_damage.mean():.3f}")
        print(f"    Windows: {window_damage.mean():.3f}")
        print(f"    Foundation: {foundation_damage.mean():.3f}")
        print(f"  Base wind damage: {base_wind_damage.min():.3f} - {base_wind_damage.max():.3f}")
        print(f"  With fatigue: {wind_damage.min():.3f} - {wind_damage.max():.3f}")
        print(f"  Total damage: {total_damage.min():.3f} - {total_damage.max():.3f} (mean: {total_damage.mean():.3f})")
        
        return {
            # Component-level damage
            'roof_damage': roof_damage,
            'wall_damage': wall_damage,
            'window_damage': window_damage,
            'foundation_damage': foundation_damage,
            
            # Aggregated damage
            'base_wind_damage': base_wind_damage,
            'wind_damage_with_fatigue': wind_damage,
            'debris_damage': debris_damage,
            'surge_damage': surge_damage,
            'total_damage_ratio': total_damage,
            
            # Pass through indices
            'event_indices': hazard_results['event_indices'],
            'location_indices': hazard_results['location_indices']
        }
    
    def _compute_component_damage(self, wind_speed_kt: Tensor, v50: float, beta: float) -> Tensor:
        """
        Compute damage for a single component using sigmoid vulnerability curve
        
        Args:
            wind_speed_kt: Wind speed in knots
            v50: Wind speed at 50% damage
            beta: Steepness parameter
        
        Returns:
            Damage ratio 0-1
        """
        x = (wind_speed_kt - v50) / v50
        damage = 1.0 / (1.0 + torch.exp(-beta * x))
        return damage

# =============================================================================
# KERNEL 4: FINANCIAL AGGREGATION
# =============================================================================

class FinancialAggregationKernel:
    """
    Kernel 4: Portfolio-level financial loss computation
    
    CORRECTED INSURANCE LOGIC:
    1. Ground-Up (GU): Total economic loss per event (sum across all locations)
    2. Gross Loss: Apply policy terms AT EVENT LEVEL:
       - Sum all location losses for the event
       - Subtract deductible ONCE per event
       - Cap at policy limit
    3. Year Loss Table (YLT): Aggregates events into annual buckets
    """
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.device = config.device
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        
    def aggregate_financial_loss(
        self,
        vulnerability_results: Dict[str, Tensor],
        building_values: Tensor,
        event_year_mapping: Tensor,  # [n_events] mapping event_id -> simulated_year
        policy_terms: Optional[Dict] = None,
        n_events: int = 0,
        n_sim_years: int = 1000,
        catalog_type: str = 'stochastic'
    ) -> Dict[str, Tensor]:
        
        print(f"\n{'='*70}")
        print(f"KERNEL 4: INSURANCE-GRADE FINANCIAL AGGREGATION")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        event_year_mapping = event_year_mapping.to(self.device).long()
        
        # 1. EXTRACT DATA
        damage_ratios = vulnerability_results['total_damage_ratio']
        event_indices = vulnerability_results['event_indices']
        location_indices = vulnerability_results['location_indices']
        
        # 2. CALCULATE PAIR-LEVEL GROUND-UP LOSSES
        
        affected_values = building_values[location_indices]
        gu_loss_pair = damage_ratios * affected_values
        
        # 3. AGGREGATE TO EVENT LEVEL - GROUND UP
        # Sum all location losses for each event
        event_gu = torch.zeros(n_events, device=self.device, dtype=self.config.dtype)
        event_gu.scatter_add_(0, event_indices, gu_loss_pair)
        
        # 4. APPLY INSURANCE POLICY TERMS AT EVENT LEVEL
        # This is the key correction: deductible and limit apply to the EVENT, not each location
        if policy_terms is not None:
            deductible = policy_terms.get('deductible', 0.0)
            limit = policy_terms.get('limit', float('inf'))
            
            # For each event:
            # 1. Sum all location losses
            # 2. Subtract deductible ONCE
            # 3. Cap at limit
            event_gross = torch.clamp(event_gu - deductible, min=0.0, max=limit)
            
            print(f"\n  Policy Terms Applied at Event Level:")
            print(f"    Deductible: ${deductible:,.0f} per event")
            print(f"    Limit: ${limit:,.0f} per event")
        else:
            # No policy terms - gross equals ground-up
            event_gross = event_gu.clone()
            print(f"\n  No policy terms - using ground-up losses")
        
        # 5. AGGREGATE TO ANNUAL LEVEL (Year Loss Table - YLT)
        annual_gu = torch.zeros(n_sim_years, device=self.device, dtype=self.config.dtype)
        annual_gross = torch.zeros(n_sim_years, device=self.device, dtype=self.config.dtype)
        
        # Sum all events in each year
        annual_gu.index_add_(0, event_year_mapping, event_gu)
        annual_gross.index_add_(0, event_year_mapping, event_gross)
        
        # 6. RISK METRIC CALCULATIONS
        # Expected Annual Loss (AAL) is the mean of the annual losses
        aal_gu = annual_gu.mean().item()
        aal_gross = annual_gross.mean().item()
        
        # Volatility metrics
        std_dev_gross = annual_gross.std().item()
        cv_gross = std_dev_gross / aal_gross if aal_gross > 0 else 0.0
        
        # Total Portfolio Value
        tiv = building_values.sum().item()
        
        # 7. EXCEEDANCE PROBABILITY (EP) CURVE
        sorted_annual_losses, _ = torch.sort(annual_gross, descending=True)
        exceedance_probs = torch.arange(1, n_sim_years + 1, device=self.device, dtype=self.config.dtype) / n_sim_years
        
        # 8. PROBABLE MAXIMUM LOSS (PML)
        def calculate_pml(rp):
            idx = max(0, int(n_sim_years / rp) - 1)
            return sorted_annual_losses[idx].item()

        pml_metrics = {
            'pml_20yr': calculate_pml(20),
            'pml_50yr': calculate_pml(50),
            'pml_100yr': calculate_pml(100),
            'pml_250yr': calculate_pml(250),
            'pml_500yr': calculate_pml(500)
        }
        
        # 9. TAIL VALUE AT RISK (TVaR)
        tvar_100yr = sorted_annual_losses[:int(n_sim_years/100)].mean().item()
        
        # 10. ANALYZE POLICY IMPACT
        if policy_terms is not None:
            # Calculate how much the policy reduces loss
            total_gu_loss = event_gu.sum().item()
            total_gross_loss = event_gross.sum().item()
            policy_reduction = total_gu_loss - total_gross_loss
            
            # Count events affected by deductible and limit
            events_with_gu_loss = (event_gu > 0).sum().item()
            events_exceed_deductible = (event_gu > deductible).sum().item()
            events_capped_by_limit = (event_gu > deductible + limit).sum().item()
            
            print(f"\n  Policy Impact Analysis:")
            print(f"    Events with GU loss: {events_with_gu_loss:,}")
            print(f"    Events exceeding deductible: {events_exceed_deductible:,}")
            print(f"    Events capped by limit: {events_capped_by_limit:,}")
            print(f"    Total GU loss: ${total_gu_loss/1e9:.2f}B")
            print(f"    Total Gross loss: ${total_gross_loss/1e9:.2f}B")
            print(f"    Policy saves: ${policy_reduction/1e9:.2f}B ({100*policy_reduction/total_gu_loss:.1f}%)")
        
        # Run standard sanity checks
        self._run_sanity_checks(aal_gross, pml_metrics, tiv, catalog_type)
        
        agg_time = time.time() - start_time
        
        print(f"\n  Aggregation complete in {agg_time:.2f}s")
        print(f"  Portfolio TIV: ${tiv/1e9:.2f}B")
        print(f"  Ground-Up AAL: ${aal_gu/1e6:.2f}M")
        print(f"  Gross (Insured) AAL: ${aal_gross/1e6:.2f}M")
        print(f"  Annual Std Dev: ${std_dev_gross/1e6:.2f}M")
        print(f"  Coefficient of Variation (CV): {cv_gross:.2f}")
        print(f"  Loss Cost (Burn Rate): {(aal_gross/tiv)*10000:.1f} bps")
        
        return {
            'expected_annual_loss': aal_gross,
            'expected_annual_loss_gu': aal_gu,
            'annual_loss_std': std_dev_gross,
            'cv': cv_gross,
            'pml_metrics': pml_metrics,
            'tvar_100yr': tvar_100yr,
            'annual_losses': annual_gross,
            'aep_curve': {'losses': sorted_annual_losses, 'probs': exceedance_probs},
            'n_events_with_loss': (event_gross > 0).sum().item(),
            'max_event_loss': event_gross.max().item(),
            'event_ground_up_losses': event_gu,
            'event_insured_losses': event_gross
        }
    
    def _run_sanity_checks(self, aal, pml_metrics, tiv, catalog_type):
        """Standard actuarial validation checks"""
        print(f"\n  {'='*66}")
        print(f"  SANITY CHECKS")
        print(f"  {'='*66}")
        
        # Check 1: Loss Cost
        loss_cost = aal / tiv
        print(f"  1. Burn Rate: {loss_cost*100:.2f}%")
        if loss_cost > 0.10: 
            print("     ❌ ERROR: Loss cost > 10% (Extreme Risk/Calibration Error)")
        elif loss_cost < 0.0001: 
            print("     ⚠️  WARNING: Loss cost extremely low")
        else: 
            print("     ✓ PASS: Burn rate in plausible range")

        # Check 2: PML Monotonicity
        pmls = list(pml_metrics.values())
        is_monotonic = all(x <= y for x, y in zip(pmls, pmls[1:]))
        print(f"  2. Monotonicity: {'✓ PASS' if is_monotonic else '❌ ERROR: Non-monotonic PML curve'}")
        
        # Check 3: AAL vs 100yr PML
        ratio = aal / pml_metrics['pml_100yr'] if pml_metrics['pml_100yr'] > 0 else 0
        print(f"  3. AAL/PML100 Ratio: {ratio:.2f}")
        if ratio < 0.01 or ratio > 0.5:
            print(f"     ⚠️  WARNING: Ratio outside typical range (0.01-0.50)")
        else:
            print(f"     ✓ PASS: Ratio in expected range")


# =============================================================================
# INTEGRATED FOUR-KERNEL PIPELINE
# =============================================================================

class TensorCatPipeline:
    """
    Complete TensorCat four-kernel pipeline
    
    Orchestrates the full GPU-native catastrophe modeling workflow
    """
    
    def __init__(self, config: Optional[KernelConfig] = None):
        if config is None:
            config = KernelConfig()
        
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        
        if not torch.cuda.is_available() and config.device == 'cuda':
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Initialize kernels
        self.spatial_filter = SpatialFilteringKernel(config)
        self.temporal_hazard = TemporalHazardStreamingKernel(config)
        self.vulnerability = VulnerabilityAssessmentKernel(config)
        self.financial_agg = FinancialAggregationKernel(config)
        
        print(f"\n{'='*70}")
        print(f"TENSORCAT FOUR-KERNEL ARCHITECTURE INITIALIZED")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Dtype: {config.dtype}")
        print(f"Mixed Precision: {config.use_mixed_precision}")
        print(f"Spatial filter radius: {config.spatial_filter_radius_km:.1f} km")
        print(f"Spatial batch size: {config.spatial_batch_size}")
        print(f"Temporal batch size: {config.temporal_batch_size}")
        print(f"Time step: {config.time_step_hours:.1f} hours")
        print(f"Fatigue modeling: {config.fatigue_enabled}")
        print(f"Debris modeling: {config.debris_enabled}")
        
    def run_pipeline(
        self,
        event_tracks: Dict[str, Tensor],
        location_coords: Tensor,
        building_values: Tensor,
        building_type: str = 'residential',
        event_occurrence_rates: Optional[Tensor] = None,
        event_year_mapping: Optional[Tensor] = None,
        policy_terms: Optional[Dict] = None, 
        n_sim_years: int = 1000 
    ) -> Dict[str, Tensor]:
        """
        Execute complete four-kernel pipeline WITH probabilistic analysis
        
        Args:
            event_tracks: Storm track data dictionary
            location_coords: [n_locations, 2] coordinates
            building_values: [n_locations] replacement values
            building_type: Building classification
            event_occurrence_rates: [n_events] annual occurrence rates (optional)
            policy_terms: Optional insurance terms
            
        Returns:
            Complete results dictionary with probabilistic outputs
        """
        pipeline_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"TENSORCAT PIPELINE EXECUTION")
        print(f"{'='*70}")
        print(f"Events: {event_tracks['n_events']:,}")
        print(f"Locations: {location_coords.shape[0]:,}")
        print(f"Building type: {building_type}")
        if event_occurrence_rates is not None:
            print(f"Probabilistic analysis: Enabled")
        
        # Move all input tensors to device
        print(f"\nTransferring data to {self.device}...")
        
        # Check GPU memory before starting
        if torch.cuda.is_available():
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            gpu_mem_free = gpu_mem_total - gpu_mem_reserved
            
            print(f"GPU Memory Status:")
            print(f"  Total: {gpu_mem_total:.2f} GB")
            print(f"  Free: {gpu_mem_free:.2f} GB")
            print(f"  Allocated: {gpu_mem_allocated:.2f} GB")
            
            # Estimate required memory
            n_potential_pairs = event_tracks['n_events'] * location_coords.shape[0]
            estimated_filtered = n_potential_pairs * 0.04  # Assume 96% filtering
            estimated_memory = estimated_filtered * 4 * 10 / 1e9  # 10 floats per pair, 4 bytes each
            
            print(f"  Estimated peak: {estimated_memory:.2f} GB")
            
            if estimated_memory > gpu_mem_free * 0.8:
                print(f"\n⚠️  WARNING: Estimated memory ({estimated_memory:.2f} GB) may exceed available ({gpu_mem_free:.2f} GB)")
                print(f"   Consider reducing location count or using CPU")
                proceed = input("   Proceed anyway? (y/n): ").strip().lower()
                if proceed != 'y':
                    return None
        
        event_tracks_device = {
            k: v.to(self.device) if isinstance(v, Tensor) else v 
            for k, v in event_tracks.items()
        }
        location_coords = location_coords.to(self.device)
        building_values = building_values.to(self.device)
        if event_occurrence_rates is not None:
            event_occurrence_rates = event_occurrence_rates.to(self.device)
        
        # KERNEL 1: Spatial Filtering
        spatial_results = self.spatial_filter.filter_events_by_location(
            event_tracks_device, location_coords
        )
        
        # KERNEL 2: Temporal Hazard Streaming
        hazard_results = self.temporal_hazard.stream_temporal_hazard(
            event_tracks_device, location_coords, spatial_results
        )
        
        # KERNEL 3: Vulnerability Assessment
        vulnerability_results = self.vulnerability.assess_vulnerability(
            hazard_results, building_type=building_type
        )
        
        # KERNEL 4: Financial Aggregation WITH probabilities
        financial_results = self.financial_agg.aggregate_financial_loss(
            vulnerability_results,
            building_values,
            event_year_mapping=event_year_mapping,
            policy_terms=policy_terms,
            n_events=event_tracks['n_events'],
            n_sim_years=n_sim_years  # Ensure this matches your catalog duration
        )
        
        pipeline_time = time.time() - pipeline_start
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {pipeline_time:.2f}s")
        print(f"Events/second: {event_tracks['n_events']/pipeline_time:.0f}")
        print(f"Locations/second: {location_coords.shape[0]/pipeline_time:.0f}")
        
        # Combine all results
        return {
            'spatial_filter': spatial_results,
            'hazard': hazard_results,
            'vulnerability': vulnerability_results,
            'financial': financial_results,
            'total_time_seconds': pipeline_time
        }
