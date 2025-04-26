#!/usr/bin/env python3
"""
Script to extract Wyscout and SkillCorner data files for Sweden's matches.
This script reads the sweden_matches.csv file, identifies the corresponding data files,
and copies them to a new location.
"""

import os
import csv
import shutil
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SWEDEN_MATCHES_CSV = DATA_DIR / "sweden_matches.csv"
WYSCOUT_DIR = DATA_DIR / "wyscout"  # Updated path
SKILLCORNER_DIR = DATA_DIR / "skillcorner"  # Updated path

# Create directory for extracted data
OUTPUT_DIR = DATA_DIR / "sweden_data"  # Updated path
OUTPUT_WYSCOUT_DIR = OUTPUT_DIR / "wyscout"
OUTPUT_SKILLCORNER_DIR = OUTPUT_DIR / "skillcorner"

def ensure_dir(directory):
    """Ensure the directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_match_data():
    """Extract Wyscout and SkillCorner data files for Sweden's matches."""
    # Ensure output directories exist
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_WYSCOUT_DIR)
    ensure_dir(OUTPUT_SKILLCORNER_DIR)
    
    print(f"Looking for Wyscout files in: {WYSCOUT_DIR}")
    print(f"Looking for SkillCorner files in: {SKILLCORNER_DIR}")
    
    # Read Sweden matches from CSV
    matches = []
    with open(SWEDEN_MATCHES_CSV, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            matches.append(row)
    
    # Lists to track successfully extracted files
    wyscout_files_found = []
    skillcorner_files_found = []
    
    # Process each match
    for match in matches:
        wyscout_id = match['wyscout']
        skillcorner_id = match['skillcorner']
        date = match['date']
        home = match['home']
        away = match['away']
        
        # Process Wyscout files (look in the 2024 directory and other potential subdirectories)
        if wyscout_id:
            # Try to find the file in the 2024 directory first
            wyscout_file_2024 = WYSCOUT_DIR / "2024" / f"{wyscout_id}.json"
            
            # Check other directories if needed (unl2025-md1+2 and unl2025-md3+4)
            wyscout_file_unl1 = WYSCOUT_DIR / "unl2025-md1+2" / f"{wyscout_id}.json"
            wyscout_file_unl2 = WYSCOUT_DIR / "unl2025-md3+4" / f"{wyscout_id}.json"
            
            # Print locations being checked for debugging
            print(f"Checking for Wyscout file with ID {wyscout_id}:")
            print(f" - {wyscout_file_2024}")
            print(f" - {wyscout_file_unl1}")
            print(f" - {wyscout_file_unl2}")
            
            if os.path.exists(wyscout_file_2024):
                dest_file = OUTPUT_WYSCOUT_DIR / f"{wyscout_id}_{home}_vs_{away}_{date}.json"
                shutil.copy2(wyscout_file_2024, dest_file)
                wyscout_files_found.append(f"{wyscout_id} ({home} vs {away})")
                print(f" -> Found at {wyscout_file_2024}")
            elif os.path.exists(wyscout_file_unl1):
                dest_file = OUTPUT_WYSCOUT_DIR / f"{wyscout_id}_{home}_vs_{away}_{date}.json"
                shutil.copy2(wyscout_file_unl1, dest_file)
                wyscout_files_found.append(f"{wyscout_id} ({home} vs {away})")
                print(f" -> Found at {wyscout_file_unl1}")
            elif os.path.exists(wyscout_file_unl2):
                dest_file = OUTPUT_WYSCOUT_DIR / f"{wyscout_id}_{home}_vs_{away}_{date}.json"
                shutil.copy2(wyscout_file_unl2, dest_file)
                wyscout_files_found.append(f"{wyscout_id} ({home} vs {away})")
                print(f" -> Found at {wyscout_file_unl2}")
            else:
                print(f" -> NOT found in any directory")
        
        # Process SkillCorner files
        if skillcorner_id:
            skillcorner_file = SKILLCORNER_DIR / f"{skillcorner_id}.zip"
            print(f"Checking for SkillCorner file with ID {skillcorner_id}:")
            print(f" - {skillcorner_file}")
            
            if os.path.exists(skillcorner_file):
                dest_file = OUTPUT_SKILLCORNER_DIR / f"{skillcorner_id}_{home}_vs_{away}_{date}.zip"
                shutil.copy2(skillcorner_file, dest_file)
                skillcorner_files_found.append(f"{skillcorner_id} ({home} vs {away})")
                print(f" -> Found at {skillcorner_file}")
            else:
                print(f" -> NOT found")
    
    # Print summary
    print("\nExtraction complete!")
    print(f"Found and copied {len(wyscout_files_found)} Wyscout files:")
    for file_info in wyscout_files_found:
        print(f" - {file_info}")
    
    print(f"\nFound and copied {len(skillcorner_files_found)} SkillCorner files:")
    for file_info in skillcorner_files_found:
        print(f" - {file_info}")
    
    total_matches = len(matches)
    print(f"\nSummary: Processed {total_matches} Sweden matches")
    print(f" - Wyscout coverage: {len(wyscout_files_found)}/{total_matches} matches ({len(wyscout_files_found)/total_matches*100:.1f}%)")
    print(f" - SkillCorner coverage: {len(skillcorner_files_found)}/{total_matches} matches ({len(skillcorner_files_found)/total_matches*100:.1f}%)")
    print(f"\nExtracted data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_match_data()