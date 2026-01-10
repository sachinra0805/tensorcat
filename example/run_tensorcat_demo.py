"""
TensorCat Demo Script
Run this to execute the complete TensorCat pipeline

Usage:
    python run_tensorcat_demo.py

Author: Sachin Ramnath Arunkumar
Date: January 2026
"""

import numpy as np
import torch
from tensorcat_core import TensorCatPipeline, KernelConfig
from tensorcat_utils import load_storm_data


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║         TENSORCAT FOUR-KERNEL ARCHITECTURE DEMO                   ║
║         GPU-Native Catastrophe Modeling                           ║
║                                                                   ║
║  Kernel 1: Spatial Filtering                                      ║
║  Kernel 2: Temporal Hazard Streaming                              ║
║  Kernel 3: Vulnerability Assessment                               ║
║  Kernel 4: Financial Aggregation                                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # OPTIMIZATION PRESETS
    # =========================================================================
    
    print("\nSelect optimization preset:")
    print("1. STANDARD (up to 10K events) - Conservative settings")
    print("2. OPTIMIZED (10K-50K events) - Larger batches, mixed precision")
    print("3. MAXIMUM (50K+ events) - All optimizations enabled")
    print("4. AGGRESSIVE (experimental) - 4x batch sizes with auto-downgrade safety")
    print("5. CUSTOM - Specify your own settings")
    
    preset = input("\nEnter preset (1-5, default=2): ").strip() or "2"
    
    if preset == "1":
        # Standard configuration
        config = KernelConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
            spatial_filter_radius_km=150.0,
            temporal_batch_size=50,
            spatial_batch_size=500,
            use_mixed_precision=False,
            fatigue_enabled=True,
            debris_enabled=True
        )
        print("\n✓ STANDARD preset loaded")
        
    elif preset == "3":
        # Maximum optimization
        config = KernelConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
            spatial_filter_radius_km=400.0,
            temporal_batch_size=200,
            spatial_batch_size=1000,
            use_mixed_precision=True,
            fatigue_enabled=True,
            debris_enabled=True
        )
        print("\n✓ MAXIMUM preset loaded (2-3x speedup expected)")
        
    elif preset == "4":
        # AGGRESSIVE configuration with auto-downgrade safety
        config = KernelConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
            spatial_filter_radius_km=500.0,
            temporal_batch_size=400,
            spatial_batch_size=3000,
            use_mixed_precision=True,
            fatigue_enabled=True,
            debris_enabled=True
        )
        print("\n✓ AGGRESSIVE preset loaded")
        print("   ⚠️  WARNING: This config pushes GPU limits")
        print("   Auto-downgrade enabled: will reduce batch size if OOM detected")
        
    elif preset == "5":
        # Custom configuration
        print("\nCustom configuration:")
        use_fp16 = input("Use mixed precision FP16? (y/n, default=y): ").strip().lower() != 'n'
        temp_batch = int(input("Temporal batch size (default=100): ").strip() or "100")
        filter_km = float(input("Spatial filter radius km (default=500): ").strip() or "500")
        
        config = KernelConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
            spatial_filter_radius_km=filter_km,
            temporal_batch_size=temp_batch,
            spatial_batch_size=500,
            use_mixed_precision=use_fp16,
            fatigue_enabled=True,
            debris_enabled=True
        )
        print(f"\n✓ CUSTOM preset loaded")
        
    else:  # preset == "2" (default OPTIMIZED)
        # Optimized configuration (recommended for 107K events)
        config = KernelConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float32,
            spatial_filter_radius_km=500.0,
            temporal_batch_size=100,
            spatial_batch_size=1000,
            use_mixed_precision=True,
            fatigue_enabled=True,
            debris_enabled=True
        )
        print("\n✓ OPTIMIZED preset loaded (recommended for 100K+ events)")
    
    # =========================================================================
    # DATASET CONFIGURATION
    # =========================================================================
    
    # Example: Load real data (update path as needed)
    try:
        # Try to load actual storm data
        storm_data_path = "df_all.parquet"
        
        # Ask user for number of events to process
        print(f"\nStorm data file: {storm_data_path}")
        max_events_input = input("Number of events to process (default=10000, 'all' for full dataset): ").strip()
        
        if max_events_input.lower() == 'all':
            max_events = None
            print("Processing ALL events in dataset")
        else:
            max_events = int(max_events_input) if max_events_input else 10000
            print(f"Processing {max_events:,} events")
        
        event_tracks, event_occurrence_rates, event_year_mapping, n_sim_years = load_storm_data(storm_data_path, max_events=max_events)
        
        # Define portfolio location count
        print(f"\nLocation Configuration:")
        print("Examples: 100 (10×10), 400 (20×20), 1024 (32×32), 10000 (100×100)")
        loc_input = input("Number of locations (default=100): ").strip()
        
        if loc_input:
            target_locs = int(loc_input)
            # Calculate grid size (square root)
            grid_size = int(np.sqrt(target_locs))
            actual_locs = grid_size * grid_size
            
            # Safety check for large location counts
            if actual_locs > 500:
                print(f"\n⚠️  WARNING: {actual_locs} locations detected")
                print(f"   Estimated memory: {actual_locs * 107000 * 4 / 1e9 * 0.033:.1f} GB")
                
                # Automatic safety adjustments
                if actual_locs >= 1000:
                    print(f"   Applying MEMORY-SAFE configuration:")
                    config.temporal_batch_size = 50
                    config.spatial_batch_size = 500
                    config.spatial_filter_radius_km = 300.0
                    print(f"   - Temporal batch: 200 → 50")
                    print(f"   - Spatial batch: 1000 → 500")
                    print(f"   - Filter radius: 400km → 300km")
                elif actual_locs >= 500:
                    print(f"   Applying CONSERVATIVE configuration:")
                    config.temporal_batch_size = 100
                    config.spatial_batch_size = 750
                    config.spatial_filter_radius_km = 350.0
                
                confirm = input(f"\n   Continue with {actual_locs} locations? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Exiting...")
                    return
            
            print(f"\n✓ Using {grid_size}×{grid_size} = {actual_locs} locations")
        else:
            grid_size = 10
            actual_locs = 100
            print(f"\n✓ Using default 10×10 = 100 locations")
        
        # Define portfolio (San Jose Island, Texas area)
        lons = np.linspace(-99.0, -96.5, grid_size)
        lats = np.linspace(27.0, 28.5, grid_size)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        grid_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        location_coords = torch.tensor(grid_coords, dtype=torch.float32)
        
        # Building values (example: $1M - $10M per location)
        n_locations = location_coords.shape[0]
        building_values = torch.tensor(
            np.random.uniform(1e6, 10e6, n_locations),
            dtype=torch.float32
        )
        
        # Policy terms (example: $100k deductible, $5M limit)
        policy_terms = {
            'deductible': 100000.0,
            'limit': 5000000.0
        }
        
        # Initialize and run pipeline
        pipeline = TensorCatPipeline(config)
        
        try:
            results = pipeline.run_pipeline(
                event_tracks=event_tracks,
                location_coords=location_coords,
                building_values=building_values,
                building_type='residential',
                event_year_mapping=event_year_mapping,
                policy_terms=policy_terms,
                n_sim_years=n_sim_years
            )
            
            if results is None:
                print("\n❌ Pipeline execution cancelled by user")
                return
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n{'='*70}")
                print(f"❌ GPU OUT OF MEMORY ERROR")
                print(f"{'='*70}")
                print(f"\nYour configuration exceeded available GPU memory.")
                print(f"\nRecovery options:")
                print(f"1. Reduce location count (try half: {actual_locs//2})")
                print(f"2. Reduce event count")
                print(f"3. Use preset 1 (STANDARD) with smaller batches")
                print(f"4. Tighten spatial filter radius to 250km")
                print(f"5. Switch to CPU (will be slower)")
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                print(f"\n💡 Tip: For {actual_locs} locations, recommended max events: ~50,000")
                return
            else:
                raise e
        
        # Display results
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\nSpatial Filtering:")
        print(f"  Filtered pairs: {results['spatial_filter']['n_filtered_pairs']:,}")
        print(f"  Reduction: {100*(1 - results['spatial_filter']['n_filtered_pairs']/(event_tracks['n_events']*n_locations)):.1f}%")
        
        print(f"\nHazard:")
        print(f"  Peak wind range: {results['hazard']['peak_wind_speed'].min():.1f} - {results['hazard']['peak_wind_speed'].max():.1f} m/s")
        print(f"  Mean exposure: {results['hazard']['exposure_duration'].mean():.1f} hours")
        
        print(f"\nVulnerability:")
        print(f"  Mean damage ratio: {results['vulnerability']['total_damage_ratio'].mean():.3f}")
        print(f"  Max damage ratio: {results['vulnerability']['total_damage_ratio'].max():.3f}")
        
        print(f"\nFinancial:")
        print(f"  Events with loss: {results['financial']['n_events_with_loss']:,}")
        print(f"  Max event loss: ${results['financial']['max_event_loss']/1e6:.2f}M")
        
        # Display probabilistic risk metrics if available
        if 'expected_annual_loss' in results['financial'] and results['financial']['expected_annual_loss'] > 0:
            eal = results['financial']['expected_annual_loss']
            print(f"  Expected Annual Loss (EAL): ${eal/1e6:.2f}M")
            
            if 'pml_metrics' in results['financial'] and results['financial']['pml_metrics']:
                pml = results['financial']['pml_metrics']
                print(f"\n  Probable Maximum Loss (PML):")
                if 'pml_100yr' in pml:
                    print(f"    100-year: ${pml['pml_100yr']/1e6:.2f}M")
                if 'pml_250yr' in pml:
                    print(f"    250-year: ${pml['pml_250yr']/1e6:.2f}M")
                if 'pml_500yr' in pml:
                    print(f"    500-year: ${pml['pml_500yr']/1e6:.2f}M")
        else:
            print(f"  Total portfolio loss sum: ${results['financial']['event_insured_losses'].sum().item()/1e9:.2f}B")
        
        # Performance Analysis
        print(f"\n{'='*70}")
        print(f"PERFORMANCE ANALYSIS")
        print(f"{'='*70}")
        total_time = results['total_time_seconds']
        n_events = event_tracks['n_events']
        n_pairs = results['spatial_filter']['n_filtered_pairs']
        
        print(f"\nThroughput:")
        print(f"  Events/second: {n_events/total_time:.0f}")
        print(f"  Pairs/second: {n_pairs/total_time:.0f}")
        print(f"  Total timesteps processed: {n_pairs * event_tracks['max_timesteps']:,}")
        print(f"  Timestep evaluations/second: {(n_pairs * event_tracks['max_timesteps'])/total_time:.0f}")
        
        print(f"\nOptimizations Applied:")
        print(f"  Mixed precision (FP16): {config.use_mixed_precision}")
        print(f"  Temporal batch size: {config.temporal_batch_size}")
        print(f"  Spatial batch size: {config.spatial_batch_size}")
        
        print(f"\nEstimated Performance vs Traditional:")
        trad_time = n_events * 2.0  # Traditional models: ~2 seconds per event
        speedup = trad_time / total_time
        print(f"  Traditional cat model time: ~{trad_time/60:.1f} minutes")
        print(f"  TensorCat time: {total_time:.1f} seconds")
        print(f"  Speedup: {speedup:.0f}x faster")
        
    except FileNotFoundError:
        print(f"\nStorm data file not found. Using synthetic data for demo...")
        
        # Generate synthetic data for demonstration
        n_events = 1000
        max_timesteps = 50
        n_locations = 100
        
        # Synthetic storm tracks (San Jose Island, TX area)
        event_tracks = {
            'lon': torch.randn(n_events * max_timesteps) * 2 - 97.5,
            'lat': torch.randn(n_events * max_timesteps) * 1.5 + 27.5,
            'vmax': torch.rand(n_events * max_timesteps) * 40 + 20,
            'rmax': torch.rand(n_events * max_timesteps) * 50 + 20,
            'B': torch.rand(n_events * max_timesteps) * 1.5 + 1.0,
            'speed': torch.rand(n_events * max_timesteps) * 50,
            'heading': torch.rand(n_events * max_timesteps) * 2 * np.pi,
            'valid_mask': torch.ones(n_events * max_timesteps, dtype=torch.bool),
            'n_events': n_events,
            'max_timesteps': max_timesteps
        }
        
        # Create event_year_mapping for synthetic data
        event_year_mapping = torch.randint(0, 1000, (n_events,), dtype=torch.long)
        n_sim_years = 1000
        
        # Synthetic locations
        location_coords = torch.rand(n_locations, 2)
        location_coords[:, 0] = location_coords[:, 0] * 2.5 - 99.0
        location_coords[:, 1] = location_coords[:, 1] * 1.5 + 27.0
        
        building_values = torch.rand(n_locations) * 9e6 + 1e6
        
        policy_terms = {
            'deductible': 100000.0,
            'limit': 5000000.0
        }
        
        # Run pipeline
        pipeline = TensorCatPipeline(config)
        results = pipeline.run_pipeline(
            event_tracks=event_tracks,
            location_coords=location_coords,
            building_values=building_values,
            building_type='residential',
            event_year_mapping=event_year_mapping,
            policy_terms=policy_terms,
            n_sim_years=n_sim_years
        )
        
        print(f"\n(Results from synthetic data demo)")


if __name__ == "__main__":
    main()
