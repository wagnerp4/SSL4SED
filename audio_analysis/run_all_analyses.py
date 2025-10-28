#!/usr/bin/env python3
"""
DESED Dataset Analysis Runner

This script runs all three analysis scripts in sequence to provide
a comprehensive analysis of the DESED dataset.
"""

import subprocess
import sys
from pathlib import Path

def run_analysis_script(script_name):
    """Run an analysis script and handle errors."""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("SUCCESS: Script completed successfully")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print("ERROR: Script failed")
            if result.stderr:
                print("\nError output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: Script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run script - {e}")
        return False
    
    return True

def main():
    """Run all analysis scripts in sequence."""
    print("DESED Dataset Comprehensive Analysis")
    print("=" * 60)
    
    scripts = [
        "dataset_summary.py",
        "event_frequency_analysis.py", 
        "event_overlap_correlation.py"
    ]
    
    results = {}
    
    for script in scripts:
        success = run_analysis_script(script)
        results[script] = success
        
        if not success:
            print(f"\nWARNING: {script} failed. Continuing with remaining scripts...")
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{script:30}: {status}")
    
    successful_scripts = sum(results.values())
    total_scripts = len(results)
    
    print(f"\nCompleted {successful_scripts}/{total_scripts} analyses successfully")
    
    if successful_scripts == total_scripts:
        print("\nAll analyses completed successfully!")
        print("Check the audio_analysis/ directory for output files and plots.")
    else:
        print(f"\n{total_scripts - successful_scripts} analysis(es) failed.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()

