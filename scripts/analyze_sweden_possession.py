#!/usr/bin/env python3
"""
Analyze possession statistics from Sweden's matches based on Wyscout data.
This script extracts possession data from match files and visualizes the results.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Path to the Sweden data directory
BASE_DIR = Path(__file__).parent.parent
SWEDEN_DATA_DIR = BASE_DIR / "data" / "sweden_data"
WYSCOUT_DIR = SWEDEN_DATA_DIR / "wyscout"
OUTPUT_DIR = SWEDEN_DATA_DIR / "analysis"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_possession_stats():
    """Analyze possession statistics from Sweden's matches."""
    match_files = list(WYSCOUT_DIR.glob("*.json"))
    
    if not match_files:
        print("No match files found in", WYSCOUT_DIR)
        return
    
    possession_stats = []
    
    for match_file in match_files:
        # Extract match information from filename
        filename = match_file.name
        match_info = filename.replace('.json', '').split('_')
        match_id = match_info[0]
        
        # Determine teams from filename
        if "Sweden_vs_" in filename:
            home_team = "Sweden"
            away_team = filename.split("Sweden_vs_")[1].split("_")[0]
        elif "_vs_Sweden" in filename:
            home_team = filename.split("_vs_Sweden")[0].split("_")[-1]
            away_team = "Sweden"
        else:
            print(f"Cannot determine teams from filename: {filename}")
            continue
        
        # Extract date from filename
        match_date = match_info[-1]
        
        # Parse the match data
        with open(match_file, 'r') as f:
            match_data = json.load(f)
        
        # Calculate possession data
        possession_data = calculate_possession(match_data, home_team, away_team, match_date)
        if possession_data:
            possession_stats.append(possession_data)
    
    if not possession_stats:
        print("No possession data could be calculated.")
        return
    
    # Convert to DataFrame
    possession_df = pd.DataFrame(possession_stats)
    
    # Sort by date
    possession_df['match_date'] = pd.to_datetime(possession_df['match_date'])
    possession_df = possession_df.sort_values('match_date')
    
    # Save to CSV
    possession_df.to_csv(OUTPUT_DIR / "possession_stats.csv", index=False)
    
    # Generate visualizations
    visualize_possession_stats(possession_df)
    
    return possession_df

def calculate_possession(match_data, home_team, away_team, match_date):
    """Calculate possession statistics from match data."""
    # Get team information from the first possession event
    team_mapping = {}
    sweden_id = None
    opponent_id = None
    
    # Initialize duration counters
    team_possession_duration = {}
    
    # Process each event in the match to calculate possession duration
    for event in match_data.get('events', []):
        if 'possession' in event and event['possession'] and 'duration' in event['possession'] and event['possession']['duration']:
            try:
                team_id = event['team']['id']
                team_name = event['team']['name']
                
                # Store team IDs and names for mapping
                if team_id not in team_mapping:
                    team_mapping[team_id] = team_name
                
                # Identify Sweden team ID
                if team_name == "Sweden":
                    sweden_id = team_id
                elif sweden_id is None and team_id != 0 and team_id is not None:
                    opponent_id = team_id
                
                # Calculate possession duration
                duration_str = event['possession']['duration']
                try:
                    duration = float(duration_str)
                    if team_id not in team_possession_duration:
                        team_possession_duration[team_id] = 0
                    team_possession_duration[team_id] += duration
                except (ValueError, TypeError):
                    # Skip invalid duration values
                    pass
            except KeyError:
                # Skip events without required data
                continue
    
    # If we didn't find valid data
    if not team_possession_duration or sweden_id is None:
        print(f"Could not calculate possession for match: {home_team} vs {away_team} ({match_date})")
        return None
    
    # Calculate total match duration from possession events
    total_duration = sum(team_possession_duration.values())
    
    if total_duration == 0:
        print(f"Zero total duration for match: {home_team} vs {away_team} ({match_date})")
        return None
    
    # Calculate possession percentages
    sweden_possession_pct = (team_possession_duration.get(sweden_id, 0) / total_duration) * 100
    opponent_possession_pct = 100 - sweden_possession_pct
    
    # Get opponent team name
    opponent_name = None
    for team_id, team_name in team_mapping.items():
        if team_id != sweden_id and team_id != 0 and team_id is not None:
            opponent_name = team_name
            break
    
    if not opponent_name:
        opponent_name = "England" if home_team != "Sweden" else away_team
    
    # Determine if Sweden was home or away
    sweden_status = "Home" if home_team == "Sweden" else "Away"
    
    # Return possession data
    return {
        'match_id': match_data['events'][0]['matchId'] if match_data['events'] else None,
        'match_date': match_date,
        'home_team': home_team,
        'away_team': away_team,
        'opponent': opponent_name,
        'sweden_status': sweden_status,
        'sweden_possession': round(sweden_possession_pct, 1),
        'opponent_possession': round(opponent_possession_pct, 1)
    }

def visualize_possession_stats(possession_df):
    """Create visualizations of possession statistics."""
    # Set up the style
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12,
        'figure.figsize': (12, 8)
    })
    
    # 1. Bar chart of possession per match
    plt.figure(figsize=(14, 8))
    
    # Create match labels
    match_labels = [f"{row['home_team']} vs {row['away_team']}\n{row['match_date']}" 
                   for _, row in possession_df.iterrows()]
    
    # Create data for stacked bar chart
    x = np.arange(len(match_labels))
    width = 0.7
    
    # Plot stacked bars
    sweden_color = '#0072C6'  # Blue
    opponent_color = '#FFDA00'  # Yellow
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    ax.bar(x, possession_df['sweden_possession'], width, label='Sweden', color=sweden_color)
    ax.bar(x, possession_df['opponent_possession'], width, bottom=possession_df['sweden_possession'], 
           label='Opponent', color=opponent_color)
    
    # Add possession percentages as text
    for i, row in enumerate(possession_df.itertuples()):
        ax.text(i, row.sweden_possession/2, f"{row.sweden_possession}%", 
                ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, row.sweden_possession + row.opponent_possession/2, f"{row.opponent_possession}%", 
                ha='center', va='center', color='black', fontweight='bold')
    
    # Customize chart
    ax.set_ylabel('Possession (%)')
    ax.set_title('Ball Possession per Match', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(match_labels, rotation=30, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "possession_per_match.png", dpi=300, bbox_inches='tight')
    
    # 2. Pie chart of average possession
    avg_sweden_possession = possession_df['sweden_possession'].mean()
    avg_opponent_possession = possession_df['opponent_possession'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sizes = [avg_sweden_possession, avg_opponent_possession]
    labels = ['Sweden', 'Opponents']
    colors = [sweden_color, opponent_color]
    explode = (0.1, 0)  # Explode the 1st slice (Sweden)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    for text in texts:
        text.set_fontsize(14)
    
    for autotext in autotexts:
        autotext.set_fontsize(14)
    
    ax.set_title('Average Possession - All Matches', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_possession.png", dpi=300, bbox_inches='tight')
    
    # 3. Home vs Away possession comparison
    home_matches = possession_df[possession_df['sweden_status'] == 'Home']
    away_matches = possession_df[possession_df['sweden_status'] == 'Away']
    
    avg_home_possession = home_matches['sweden_possession'].mean() if not home_matches.empty else 0
    avg_away_possession = away_matches['sweden_possession'].mean() if not away_matches.empty else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fix: Use separate calls for each bar with individual colors and alpha values
    bars1 = ax.bar(['Home Matches'], [avg_home_possession], color=sweden_color, alpha=1.0)
    bars2 = ax.bar(['Away Matches'], [avg_away_possession], color=sweden_color, alpha=0.7)
    
    # Combine bars for the loop
    bars = bars1.patches + bars2.patches
    
    # Add possession percentages as text
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylim(0, 100)
    ax.set_ylabel('Average Possession (%)')
    ax.set_title('Sweden\'s Average Possession - Home vs Away', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "home_away_possession.png", dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    possession_df = analyze_possession_stats()
    
    if possession_df is not None:
        print("\nPossession Statistics Summary:")
        print("-" * 50)
        print(f"Total matches analyzed: {len(possession_df)}")
        print(f"Average possession for Sweden: {possession_df['sweden_possession'].mean():.1f}%")
        print(f"Average possession for opponents: {possession_df['opponent_possession'].mean():.1f}%")
        
        # Home vs Away comparison
        home_matches = possession_df[possession_df['sweden_status'] == 'Home']
        away_matches = possession_df[possession_df['sweden_status'] == 'Away']
        
        if not home_matches.empty:
            print(f"\nHome matches ({len(home_matches)}):")
            print(f"  Average possession: {home_matches['sweden_possession'].mean():.1f}%")
        
        if not away_matches.empty:
            print(f"\nAway matches ({len(away_matches)}):")
            print(f"  Average possession: {away_matches['sweden_possession'].mean():.1f}%")
        
        print("\nMatch-by-match possession:")
        for _, row in possession_df.iterrows():
            print(f"  {row['match_date']} - {row['home_team']} vs {row['away_team']}: " +
                  f"Sweden {row['sweden_possession']}%, {row['opponent']} {row['opponent_possession']}%")