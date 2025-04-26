#!/usr/bin/env python3
"""
Script to extract all matches where Sweden played from the matches.csv file.
This extracts matches where Sweden appears in either the home or away column.
"""

import csv
import os
from pathlib import Path

# Path to the data directory relative to the script location
DATA_DIR = Path(__file__).parent.parent / "data"
MATCHES_CSV = DATA_DIR / "matches.csv"
OUTPUT_FILE = DATA_DIR / "sweden_matches.csv"

def extract_sweden_matches():
    """Extract all matches where Sweden played as either home or away team."""
    sweden_matches = []
    
    # Read the matches.csv file
    with open(MATCHES_CSV, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Get the header row
        header = next(reader)
        sweden_matches.append(header)
        
        # Get the indices for home and away columns
        home_idx = header.index('home')
        away_idx = header.index('away')
        
        # Extract rows where Sweden is either home or away
        for row in reader:
            if len(row) > max(home_idx, away_idx):  # Ensure row has enough columns
                if row[home_idx] == 'Sweden' or row[away_idx] == 'Sweden':
                    sweden_matches.append(row)
    
    # Write the extracted matches to a new CSV file
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sweden_matches)
    
    print(f"Extracted {len(sweden_matches) - 1} matches where Sweden played.")
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Also print the matches to the console
    print("\nSweden's matches:")
    print("-" * 80)
    for match in sweden_matches[1:]:  # Skip header
        home = match[home_idx]
        away = match[away_idx]
        date = match[header.index('date')]
        result = match[header.index('result')]
        
        if home == 'Sweden':
            print(f"{date}: Sweden vs {away} - {result}")
        else:
            print(f"{date}: {home} vs Sweden - {result}")
    
if __name__ == "__main__":
    extract_sweden_matches()