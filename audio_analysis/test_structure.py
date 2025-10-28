#!/usr/bin/env python3
"""
Simple test script to verify DESED dataset structure
without requiring external dependencies.
"""

import os
from pathlib import Path

def test_desed_structure():
    """Test if DESED dataset structure is correct."""
    desed_path = Path("/Users/sophon/Software/Python/SSL_SED/DESED")
    
    print("Testing DESED dataset structure...")
    print(f"DESED path: {desed_path}")
    print(f"Exists: {desed_path.exists()}")
    
    if not desed_path.exists():
        print("ERROR: DESED directory not found!")
        return False
    
    # Check annotations directory
    annotations_path = desed_path / "annotations"
    print(f"\nAnnotations directory: {annotations_path.exists()}")
    
    if annotations_path.exists():
        annotation_files = list(annotations_path.glob("*.tsv"))
        print(f"Found {len(annotation_files)} annotation files:")
        for file in annotation_files:
            print(f"  - {file.name}")
    
    # Check audio directory
    audio_path = desed_path / "audio"
    print(f"\nAudio directory: {audio_path.exists()}")
    
    if audio_path.exists():
        audio_dirs = [d for d in audio_path.iterdir() if d.is_dir()]
        print(f"Found {len(audio_dirs)} audio subdirectories:")
        for dir in audio_dirs:
            wav_count = len(list(dir.rglob("*.wav")))
            print(f"  - {dir.name}: {wav_count} wav files")
    
    print("\nDataset structure test completed successfully!")
    return True

if __name__ == "__main__":
    test_desed_structure()

