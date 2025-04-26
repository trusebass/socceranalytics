#!/usr/bin/env python3
"""
Analyze defensive metrics from Sweden's matches based on Wyscout event data.
This script calculates PPDA (Passes Per Defensive Action) and other defensive metrics
and visualizes the results, including pitch heatmaps and outcome analysis.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import csv
from mplsoccer import Pitch, VerticalPitch

# Path to the Sweden data directory
BASE_DIR = Path(__file__).parent.parent
SWEDEN_DATA_DIR = BASE_DIR / "data" / "sweden_data"
WYSCOUT_DIR = SWEDEN_DATA_DIR / "wyscout"
OUTPUT_DIR = SWEDEN_DATA_DIR / "analysis" / "defensive_metrics"
SWEDEN_MATCHES_CSV = BASE_DIR / "data" / "sweden_matches.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette for Sweden
SWEDEN_COLORS = {
    'primary': '#006AA7',    # Blue
    'secondary': '#FECC00',  # Yellow
    'text': '#212121',       # Dark gray
    'background': '#F5F5F5', # Light gray
}

def get_match_results():
    """Read match results from sweden_matches.csv."""
    match_results = {}
    with open(SWEDEN_MATCHES_CSV, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Create a unique match identifier
            match_key = f"{row['wyscout']}_{row['date']}"
            
            # Parse the result
            result = row['result']
            home_team = row['home']
            away_team = row['away']
            
            if result and result != 'missing':
                goals = result.split('-')
                home_goals = int(goals[0])
                away_goals = int(goals[1])
                
                # Determine if Sweden won, drew, or lost
                if home_team == 'Sweden':
                    if home_goals > away_goals:
                        outcome = 'win'
                    elif home_goals < away_goals:
                        outcome = 'loss'
                    else:
                        outcome = 'draw'
                else:  # Sweden was away
                    if away_goals > home_goals:
                        outcome = 'win'
                    elif away_goals < home_goals:
                        outcome = 'loss'
                    else:
                        outcome = 'draw'
                
                # Store result information
                match_results[match_key] = {
                    'result': result,
                    'outcome': outcome,
                    'sweden_goals': home_goals if home_team == 'Sweden' else away_goals,
                    'opponent_goals': away_goals if home_team == 'Sweden' else home_goals,
                    'home_team': home_team,
                    'away_team': away_team
                }
    
    return match_results

def analyze_defensive_metrics():
    """Analyze defensive metrics from Sweden's matches."""
    match_files = list(WYSCOUT_DIR.glob("*.json"))
    
    if not match_files:
        print("No match files found in", WYSCOUT_DIR)
        return
    
    # Get match results
    match_results = get_match_results()
    
    defensive_stats = []
    ppda_by_zone = []
    ppda_timeline = []
    pressing_heatmap_data = []
    
    for match_file in match_files:
        # Extract match information from filename
        filename = match_file.name
        match_info = filename.replace('.json', '').split('_')
        match_id = match_info[0]
        
        # Determine teams from filename
        if "Sweden_vs_" in filename:
            home_team = "Sweden"
            away_team = filename.split("Sweden_vs_")[1].split("_")[0]
            sweden_status = "Home"
        elif "_vs_Sweden" in filename:
            home_team = filename.split("_vs_Sweden")[0].split("_")[-1]
            away_team = "Sweden"
            sweden_status = "Away"
        else:
            print(f"Cannot determine teams from filename: {filename}")
            continue
        
        # Extract date from filename
        match_date = match_info[-1]
        
        # Create match key for result lookup
        match_key = f"{match_id}_{match_date}"
        
        # Parse the match data
        with open(match_file, 'r') as f:
            match_data = json.load(f)
        
        # Calculate PPDA metrics
        match_ppda_data = calculate_ppda_metrics(match_data, home_team, away_team, sweden_status, match_date)
        if match_ppda_data:
            # Add match result data if available
            if match_key in match_results:
                match_ppda_data['overall']['result'] = match_results[match_key]['result']
                match_ppda_data['overall']['outcome'] = match_results[match_key]['outcome']
                match_ppda_data['overall']['sweden_goals'] = match_results[match_key]['sweden_goals']
                match_ppda_data['overall']['opponent_goals'] = match_results[match_key]['opponent_goals']
            
            defensive_stats.append(match_ppda_data['overall'])
            ppda_by_zone.extend(match_ppda_data['zones'])
            ppda_timeline.extend(match_ppda_data['timeline'])
            pressing_heatmap_data.append(match_ppda_data['heatmap_data'])
    
    if not defensive_stats:
        print("No defensive data could be calculated.")
        return
    
    # Convert to DataFrame
    defensive_df = pd.DataFrame(defensive_stats)
    zones_df = pd.DataFrame(ppda_by_zone)
    timeline_df = pd.DataFrame(ppda_timeline)
    
    # Sort by date
    defensive_df['match_date'] = pd.to_datetime(defensive_df['match_date'])
    defensive_df = defensive_df.sort_values('match_date')
    
    # Save to CSV
    defensive_df.to_csv(OUTPUT_DIR / "ppda_stats.csv", index=False)
    zones_df.to_csv(OUTPUT_DIR / "ppda_by_zone.csv", index=False)
    timeline_df.to_csv(OUTPUT_DIR / "ppda_timeline.csv", index=False)
    
    # Generate visualizations
    visualize_ppda_metrics(defensive_df, zones_df, timeline_df)
    
    # Generate pitch visualizations
    visualize_pressing_zones(pressing_heatmap_data)
    
    # Analyze relationship between PPDA and match outcomes
    analyze_ppda_vs_outcome(defensive_df)
    
    return defensive_df, zones_df, timeline_df

def identify_team_ids(match_data: Dict) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Identify team IDs and Sweden's team ID from match data."""
    sweden_id = None
    opponent_id = None
    opponent_name = None
    
    # Go through events to find team information
    for event in match_data.get('events', []):
        if 'team' in event and 'id' in event['team'] and 'name' in event['team']:
            team_id = event['team']['id']
            team_name = event['team']['name']
            
            if team_name == "Sweden":
                sweden_id = team_id
            elif team_id != 0 and team_id is not None:
                opponent_id = team_id
                opponent_name = team_name
                
            # Once we have both IDs, we can break
            if sweden_id is not None and opponent_id is not None:
                break
    
    return sweden_id, opponent_id, opponent_name

def is_defensive_action(event: Dict) -> bool:
    """Check if an event is a defensive action (tackle, interception, duel, etc.)."""
    # Check for ground duels
    if event.get('groundDuel') is not None:
        return True
    
    # Check for aerial duels
    if event.get('aerialDuel') is not None:
        return True
    
    # Check for interceptions, tackles, clearances, etc. in the primary type
    if 'type' in event and 'primary' in event['type']:
        primary_type = event['type']['primary']
        defensive_types = ['interception', 'tackle', 'clearance', 'recovery', 'foul']
        if primary_type in defensive_types:
            return True
    
    # Check for interceptions, tackles, clearances, etc. in the secondary type
    if 'type' in event and 'secondary' in event['type'] and event['type']['secondary']:
        secondary_types = event['type']['secondary']
        defensive_secondary_types = ['interception', 'clearance', 'recovery', 'defensive']
        for defensive_type in defensive_secondary_types:
            if defensive_type in secondary_types:
                return True
    
    return False

def is_pass(event: Dict) -> bool:
    """Check if an event is a pass."""
    if 'type' in event and 'primary' in event['type']:
        return event['type']['primary'] == 'pass'
    return False

def get_pitch_zone(x: float, y: float) -> str:
    """
    Determine the pitch zone based on (x, y) coordinates.
    Divides the pitch into three zones along the length (defensive, middle, attacking).
    """
    # Assuming coordinates are in range 0-100
    if x < 33.3:
        return "defensive_third"
    elif x < 66.6:
        return "middle_third"
    else:
        return "attacking_third"

def get_match_period_minutes(event: Dict) -> float:
    """Convert match period and minute to a continuous timeline in minutes."""
    period = event.get('matchPeriod', '')
    minute = event.get('minute', 0)
    
    if period == '2H':
        return 45 + minute
    elif period == 'E1':
        return 90 + minute
    elif period == 'E2':
        return 105 + minute
    elif period == 'P':
        return 120 + minute
    else:  # Default to first half
        return minute

def calculate_ppda_metrics(
    match_data: Dict, 
    home_team: str, 
    away_team: str, 
    sweden_status: str,
    match_date: str
) -> Dict:
    """Calculate PPDA (Passes Per Defensive Action) metrics from match data."""
    # Identify team IDs
    sweden_id, opponent_id, opponent_name = identify_team_ids(match_data)
    
    if sweden_id is None or opponent_id is None:
        print(f"Could not identify team IDs for match: {home_team} vs {away_team}")
        return None
    
    match_id = match_data['events'][0]['matchId'] if match_data['events'] else None
    
    # Initialize counters
    opponent_passes = {
        'total': 0,
        'defensive_third': 0,
        'middle_third': 0,
        'attacking_third': 0
    }
    
    sweden_defensive_actions = {
        'total': 0,
        'defensive_third': 0,
        'middle_third': 0,
        'attacking_third': 0
    }
    
    # Initialize timeline data
    timeline_data = []
    current_window_start = 0
    window_size = 15  # 15-minute windows
    window_overlap = 5  # 5-minute overlap for smoother trends
    
    # Counters for the current time window
    window_passes = 0
    window_def_actions = 0
    
    # For heatmap visualization
    opponent_pass_locations = []
    sweden_defensive_locations = []
    
    # Process each event
    for event in match_data.get('events', []):
        # Skip events without a team
        if 'team' not in event or 'id' not in event['team']:
            continue
        
        team_id = event['team']['id']
        
        # Skip events without location data
        if 'location' not in event or event['location'] is None or 'x' not in event['location'] or 'y' not in event['location']:
            continue
        
        # Get coordinates
        x, y = event['location']['x'], event['location']['y']
        
        # For opponent passes, we're interested in passes in their own defensive third
        # which is Sweden's attacking third, so we need to adjust x coordinate for opponent
        if team_id == opponent_id:
            zone = get_pitch_zone(100 - x, y)  # Invert x for opponent actions
        else:
            zone = get_pitch_zone(x, y)
        
        # Timeline analysis
        event_time = get_match_period_minutes(event)
        
        # Check if this event is in a new time window
        if event_time >= current_window_start + window_size:
            # Record PPDA for the previous window if we have data
            if window_passes > 0 and window_def_actions > 0:
                window_ppda = window_passes / window_def_actions
                timeline_data.append({
                    'match_id': match_id,
                    'match_date': match_date,
                    'opponent': opponent_name,
                    'window_start': current_window_start,
                    'window_end': min(current_window_start + window_size, 120),  # Cap at 120 for extra time
                    'opponent_passes': window_passes,
                    'sweden_defensive_actions': window_def_actions,
                    'ppda': window_ppda
                })
            
            # Move to the next window with overlap
            current_window_start += window_size - window_overlap
            window_passes = 0
            window_def_actions = 0
        
        # Count opponent passes and store locations for heatmap
        if team_id == opponent_id and is_pass(event):
            opponent_passes['total'] += 1
            opponent_passes[zone] += 1
            window_passes += 1
            opponent_pass_locations.append((x, y))
        
        # Count Sweden's defensive actions and store locations for heatmap
        if team_id == sweden_id and is_defensive_action(event):
            sweden_defensive_actions['total'] += 1
            sweden_defensive_actions[zone] += 1
            window_def_actions += 1
            sweden_defensive_locations.append((x, y))
    
    # Handle the last window if it has data
    if window_passes > 0 and window_def_actions > 0:
        window_ppda = window_passes / window_def_actions
        timeline_data.append({
            'match_id': match_id,
            'match_date': match_date,
            'opponent': opponent_name,
            'window_start': current_window_start,
            'window_end': min(current_window_start + window_size, 120),
            'opponent_passes': window_passes,
            'sweden_defensive_actions': window_def_actions,
            'ppda': window_ppda
        })
    
    # Calculate PPDA for each zone
    ppda_total = opponent_passes['total'] / sweden_defensive_actions['total'] if sweden_defensive_actions['total'] > 0 else float('inf')
    ppda_def_third = opponent_passes['defensive_third'] / sweden_defensive_actions['defensive_third'] if sweden_defensive_actions['defensive_third'] > 0 else float('inf')
    ppda_mid_third = opponent_passes['middle_third'] / sweden_defensive_actions['middle_third'] if sweden_defensive_actions['middle_third'] > 0 else float('inf')
    ppda_atk_third = opponent_passes['attacking_third'] / sweden_defensive_actions['attacking_third'] if sweden_defensive_actions['attacking_third'] > 0 else float('inf')
    
    # Prepare zone-specific data
    zone_data = []
    for zone in ['defensive_third', 'middle_third', 'attacking_third']:
        zone_data.append({
            'match_id': match_id,
            'match_date': match_date,
            'opponent': opponent_name,
            'sweden_status': sweden_status,
            'zone': zone,
            'opponent_passes': opponent_passes[zone],
            'sweden_defensive_actions': sweden_defensive_actions[zone],
            'ppda': opponent_passes[zone] / sweden_defensive_actions[zone] if sweden_defensive_actions[zone] > 0 else float('inf')
        })
    
    # Return metrics with heatmap data
    return {
        'overall': {
            'match_id': match_id,
            'match_date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'opponent': opponent_name,
            'sweden_status': sweden_status,
            'opponent_passes': opponent_passes['total'],
            'sweden_defensive_actions': sweden_defensive_actions['total'],
            'ppda': ppda_total,
            'ppda_def_third': ppda_def_third,
            'ppda_mid_third': ppda_mid_third,
            'ppda_atk_third': ppda_atk_third
        },
        'zones': zone_data,
        'timeline': timeline_data,
        'heatmap_data': {
            'match_id': match_id,
            'match_date': match_date,
            'opponent': opponent_name,
            'opponent_pass_locations': opponent_pass_locations,
            'sweden_defensive_locations': sweden_defensive_locations,
            'ppda': ppda_total
        }
    }

def visualize_ppda_metrics(defensive_df: pd.DataFrame, zones_df: pd.DataFrame, timeline_df: pd.DataFrame):
    """Create visualizations of PPDA metrics."""
    # Set up the style
    sns.set(style="whitegrid", rc={"figure.figsize": (12, 8)})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12
    })
    
    # 1. Bar chart of PPDA per match
    # Create match labels
    match_labels = [f"{row['home_team']} vs {row['away_team']}\n{row['match_date']}" 
                   for _, row in defensive_df.iterrows()]
    
    # Plot data
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(np.arange(len(match_labels)), defensive_df['ppda'], color=SWEDEN_COLORS['primary'])
    
    # Add PPDA values as text
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Add a horizontal line for the average
    avg_ppda = defensive_df['ppda'].mean()
    ax.axhline(y=avg_ppda, color=SWEDEN_COLORS['secondary'], linestyle='--', 
               label=f'Average PPDA: {avg_ppda:.2f}')
    
    # Customize chart
    ax.set_ylabel('PPDA (Passes Per Defensive Action)')
    ax.set_title('Sweden\'s PPDA per Match', fontsize=16, fontweight='bold')
    ax.set_xticks(np.arange(len(match_labels)))
    ax.set_xticklabels(match_labels, rotation=45, ha='right')
    ax.legend()
    
    # Add an annotation to explain PPDA
    ax.annotate('Lower PPDA = More intense pressing', xy=(0.5, 0.01), xycoords='axes fraction', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", fc=SWEDEN_COLORS['background'], ec=SWEDEN_COLORS['text'], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppda_per_match.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PPDA by zone per match
    # Pivot the zone data for plotting
    zones_pivot = zones_df.pivot_table(
        index=['match_id', 'match_date', 'opponent'], 
        columns='zone', 
        values='ppda'
    ).reset_index()
    
    # Convert match_date to datetime for sorting
    zones_pivot['match_date'] = pd.to_datetime(zones_pivot['match_date'])
    zones_pivot = zones_pivot.sort_values('match_date')
    
    # Create labels
    zone_match_labels = [f"{row['opponent']}\n{row['match_date']}" 
                        for _, row in zones_pivot.iterrows()]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(zone_match_labels))
    width = 0.25
    
    # Plot bars for each third
    def_bars = ax.bar(x - width, zones_pivot['defensive_third'], width, 
                      label='Defensive Third', color=SWEDEN_COLORS['primary'], alpha=0.9)
    mid_bars = ax.bar(x, zones_pivot['middle_third'], width, 
                       label='Middle Third', color=SWEDEN_COLORS['primary'], alpha=0.6)
    atk_bars = ax.bar(x + width, zones_pivot['attacking_third'], width, 
                       label='Attacking Third', color=SWEDEN_COLORS['primary'], alpha=0.3)
    
    # Customize chart
    ax.set_ylabel('PPDA')
    ax.set_title('Sweden\'s PPDA by Zone per Match', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(zone_match_labels, rotation=45, ha='right')
    ax.legend()
    
    # Add an annotation to explain PPDA
    ax.annotate('Lower PPDA = More intense pressing', xy=(0.5, 0.01), xycoords='axes fraction', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", fc=SWEDEN_COLORS['background'], ec=SWEDEN_COLORS['text'], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppda_by_zone.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Home vs Away PPDA comparison
    home_matches = defensive_df[defensive_df['sweden_status'] == 'Home']
    away_matches = defensive_df[defensive_df['sweden_status'] == 'Away']
    
    avg_home_ppda = home_matches['ppda'].mean() if not home_matches.empty else 0
    avg_away_ppda = away_matches['ppda'].mean() if not away_matches.empty else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bars - don't use list for alpha, use single values for each bar
    home_bar = ax.bar(['Home Matches'], [avg_home_ppda], color=SWEDEN_COLORS['primary'], alpha=1.0)
    away_bar = ax.bar(['Away Matches'], [avg_away_ppda], color=SWEDEN_COLORS['primary'], alpha=0.7)
    
    # Add values as text
    for bar in home_bar:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
                
    for bar in away_bar:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Customize chart
    ax.set_ylabel('Average PPDA')
    ax.set_title('Sweden\'s Average PPDA - Home vs Away', fontsize=16, fontweight='bold')
    
    # Add an annotation to explain PPDA
    ax.annotate('Lower PPDA = More intense pressing', xy=(0.5, 0.01), xycoords='figure fraction', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc=SWEDEN_COLORS['background'], ec=SWEDEN_COLORS['text'], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "home_away_ppda.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. PPDA timeline for individual matches with bar chart and goals
    # Dummy goals data (you'll need to replace this with actual goals data)
    # For demonstration, let's create more realistic random goals (fewer per match)
    goals_data = {}
    for match_id in timeline_df['match_id'].unique():
        # Generate 1-3 random goals per match
        num_goals = np.random.randint(1, 4)
        goals = []
        
        for _ in range(num_goals):
            # Generate a random minute (between 1-90)
            minute = np.random.randint(1, 91)
            # Randomly assign to Sweden or opponent
            team = 'Sweden' if np.random.random() > 0.5 else 'Opponent'
            goals.append((minute, team))
            
        # Sort goals by minute
        goals.sort(key=lambda x: x[0])
        goals_data[match_id] = goals
    
    # Get unique matches
    unique_matches = timeline_df[['match_id', 'match_date', 'opponent']].drop_duplicates()
    
    for _, match in unique_matches.iterrows():
        match_id = match['match_id']
        match_timeline = timeline_df[timeline_df['match_id'] == match_id]
        
        # Convert window_start to a format suitable for x-axis labels
        match_timeline['window_label'] = match_timeline.apply(
            lambda x: f"{int(x['window_start'])}-{int(x['window_end'])}", axis=1
        )
        
        # Setup figure for bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot PPDA as bar chart
        bars = ax.bar(
            match_timeline['window_label'],
            match_timeline['ppda'],
            color=SWEDEN_COLORS['primary'],
            alpha=0.7,
            width=0.7
        )
        
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Add goals as vertical lines with annotations
        if match_id in goals_data and goals_data[match_id]:
            for goal_minute, team in goals_data[match_id]:
                # Find the closest window that contains this goal
                for i, row in match_timeline.iterrows():
                    if row['window_start'] <= goal_minute <= row['window_end']:
                        idx = match_timeline.index.get_loc(i)
                        if idx < len(bars):  # Make sure we don't go out of bounds
                            bar_x = bars[idx].get_x() + bars[idx].get_width()/2
                            color = SWEDEN_COLORS['primary'] if team == 'Sweden' else 'red'
                            marker = '^' if team == 'Sweden' else 'v'  # Up arrow for Sweden, down for opponent
                            
                            # Draw the goal marker
                            ax.scatter(bar_x, 0.5, s=100, marker=marker, color=color, edgecolor='black', zorder=5)
                            
                            # Add annotation
                            ax.annotate(f"Goal: {team} ({goal_minute}')", 
                                      xy=(bar_x, 0.5), xytext=(bar_x, max(match_timeline['ppda'].max() * 0.15, 1)),
                                      arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                                      ha='center', va='center', fontsize=8, fontweight='bold',
                                      bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=color, alpha=0.7))
                            break
        
        # Add average line
        avg_ppda = match_timeline['ppda'].mean()
        ax.axhline(y=avg_ppda, color=SWEDEN_COLORS['secondary'], linestyle='--', 
                  label=f'Match Avg PPDA: {avg_ppda:.2f}')
        
        # Customize plot
        ax.set_title(f"PPDA Timeline: Sweden vs {match['opponent']} ({match['match_date']})", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Match Time Period (minutes)')
        ax.set_ylabel('PPDA (lower is better)')
        
        # Add an annotation to explain PPDA
        ax.annotate('Lower PPDA = More intense pressing', xy=(0.5, 0.01), xycoords='axes fraction', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", fc=SWEDEN_COLORS['background'], ec=SWEDEN_COLORS['text'], alpha=0.7))
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ppda_timeline_{match_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. PPDA heatmap by zone
    # Calculate average PPDA by zone for each opponent
    zones_avg = zones_df.groupby(['opponent', 'zone'])['ppda'].mean().reset_index()
    zones_avg_pivot = zones_avg.pivot(index='opponent', columns='zone', values='ppda')
    
    # Setup figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        zones_avg_pivot, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd",  # Using standard colormap (not reversed)
        linewidths=.5,
        cbar_kws={'label': 'PPDA (lower is better)'}
    )
    
    plt.title('Average PPDA by Zone Against Each Opponent', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppda_zone_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. PPDA vs Opponent Passes scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    sns.scatterplot(
        data=defensive_df,
        x='opponent_passes',
        y='ppda',
        size='sweden_defensive_actions',
        sizes=(50, 300),
        hue='opponent',
        palette='colorblind'
    )
    
    # Add text labels for each point
    for _, row in defensive_df.iterrows():
        plt.text(
            row['opponent_passes'] + 5, 
            row['ppda'] + 0.1,
            row['match_date'],
            fontsize=8
        )
    
    plt.title('PPDA vs Opponent Passes', fontsize=16, fontweight='bold')
    plt.xlabel('Total Opponent Passes')
    plt.ylabel('PPDA (lower is better)')
    
    # Add annotation explaining PPDA
    plt.annotate('Lower PPDA = More intense pressing', xy=(0.5, 0.01), xycoords='figure fraction', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppda_vs_passes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR}")

def visualize_pressing_zones(heatmap_data_list):
    """Create football pitch heatmaps showing Sweden's pressing zones for each match."""
    print("Generating pitch heatmaps for pressing zones...")
    
    # Create a directory for pitch visualizations
    pitch_viz_dir = OUTPUT_DIR / "pitch_visualizations"
    os.makedirs(pitch_viz_dir, exist_ok=True)
    
    # Setup the pitch visualization
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f4f4f4', line_color='#222222', 
                 stripe_color='#f9f9f9', goal_type='box')
    
    # Collect all defensive locations for the average heatmap
    all_defensive_x = []
    all_defensive_y = []
    total_ppda = 0
    match_count = 0
    
    for heatmap_data in heatmap_data_list:
        match_id = heatmap_data['match_id']
        match_date = heatmap_data['match_date']
        opponent = heatmap_data['opponent']
        ppda = heatmap_data['ppda']
        
        # Get defensive action coordinates
        defensive_locations = heatmap_data['sweden_defensive_locations']
        
        # Skip if no data
        if not defensive_locations:
            print(f"  No defensive actions found for match against {opponent} on {match_date}")
            continue
        
        # Add to the aggregate data for average heatmap
        all_defensive_x.extend([loc[0] for loc in defensive_locations])
        all_defensive_y.extend([loc[1] for loc in defensive_locations])
        
        # Track for average PPDA calculation
        if not np.isinf(ppda):
            total_ppda += ppda
            match_count += 1
        
        # Convert to x,y arrays
        x_coords = [loc[0] for loc in defensive_locations]
        y_coords = [loc[1] for loc in defensive_locations]
        
        # Create a figure with two subplots side by side
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"Sweden's Pressing Analysis vs {opponent} ({match_date})\nPPDA: {ppda:.2f}", 
                    fontsize=18, fontweight='bold')
        
        # 1. Left plot: Scatter plot of defensive actions
        pitch.draw(ax=axs[0])
        sc = axs[0].scatter(x_coords, y_coords, c=SWEDEN_COLORS['primary'], 
                           alpha=0.7, s=40, zorder=2, edgecolors='white')
        axs[0].set_title('Sweden Defensive Actions', fontsize=16)
        
        # Add a legend for the scatter plot
        handles, labels = sc.legend_elements(prop="sizes", alpha=0.6, num=1)
        legend = axs[0].legend(handles, ['Defensive Action'], 
                             loc="upper center", bbox_to_anchor=(0.5, 0), 
                             frameon=True, ncol=1)
        
        # 2. Right plot: Heatmap of defensive actions
        pitch.draw(ax=axs[1])
        
        # Create a heatmap using kernel density estimation
        # We'll use a Gaussian kernel to create a smooth heatmap
        # Use faster histogram2d method first to create the heatmap
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, 
            bins=(20, 20),  # Use same number of bins in both dimensions
            range=[[0, 100], [0, 100]]  # Match pitch dimensions
        )
        
        # Smooth the heatmap with a Gaussian filter
        import scipy.ndimage as ndimage
        heatmap = ndimage.gaussian_filter(heatmap, sigma=2)
        
        # Create mesh grid for the contour plot with correct dimensions
        x_mesh = np.linspace(0, 100, heatmap.shape[0])
        y_mesh = np.linspace(0, 100, heatmap.shape[1])
        x_grid, y_grid = np.meshgrid(x_mesh, y_mesh)
        
        # Draw the contour plot
        contour = axs[1].contourf(
            x_grid, y_grid, heatmap.T,  # Transpose for correct orientation
            levels=15,  # Number of contour levels
            cmap='YlOrRd',  # Yellow to Red colormap
            alpha=0.8,
            zorder=1
        )
        
        axs[1].set_title('Sweden Pressing Intensity Zones', fontsize=16)
        
        # Add a colorbar to show the intensity scale
        cbar = plt.colorbar(contour, ax=axs[1])
        cbar.set_label('Pressing Intensity', fontsize=12)
        
        # Add the thirds of the pitch for reference
        for ax in axs:
            ax.axvline(x=33.3, color='gray', linestyle='--', alpha=0.7, zorder=1)
            ax.axvline(x=66.6, color='gray', linestyle='--', alpha=0.7, zorder=1)
            
            # Add text labels for each third
            ax.text(16.5, 5, "Defensive Third", color='gray', fontsize=10, ha='center')
            ax.text(50, 5, "Middle Third", color='gray', fontsize=10, ha='center')
            ax.text(83.5, 5, "Attacking Third", color='gray', fontsize=10, ha='center')
        
        # Annotations for PPDA interpretation
        for ax in axs:
            ax.annotate('Lower PPDA = More intense pressing', 
                      xy=(0.5, 0.02), xycoords='axes fraction', 
                      ha='center', va='bottom', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", fc=SWEDEN_COLORS['background'], 
                              ec=SWEDEN_COLORS['text'], alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(pitch_viz_dir / f"pressing_zones_{match_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Create a pitch visualization with pass network
        # This shows both opponent passes (which Sweden is trying to prevent)
        # and Sweden's defensive actions
        opponent_pass_locations = heatmap_data['opponent_pass_locations']
        
        if opponent_pass_locations:
            # Convert opponent pass coordinates
            opponent_x = [100 - loc[0] for loc in opponent_pass_locations]  # Flip for visualization
            opponent_y = [loc[1] for loc in opponent_pass_locations]
            
            # Setup the figure
            fig, ax = plt.subplots(figsize=(12, 8))
            pitch.draw(ax=ax)
            
            # Plot opponent passes
            ax.scatter(opponent_x, opponent_y, c='red', alpha=0.5, s=30, 
                     label='Opponent Passes', zorder=2)
            
            # Plot Sweden's defensive actions
            ax.scatter(x_coords, y_coords, c=SWEDEN_COLORS['primary'], alpha=0.7, s=50, 
                     label='Sweden Defensive Actions', zorder=3, edgecolors='white')
            
            # Add title and legend
            ax.set_title(f'Passing & Pressing Map: Sweden vs {opponent} ({match_date})\nPPDA: {ppda:.2f}', 
                       fontsize=16, fontweight='bold')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
            
            # Add the thirds of the pitch for reference
            ax.axvline(x=33.3, color='gray', linestyle='--', alpha=0.7, zorder=1)
            ax.axvline(x=66.6, color='gray', linestyle='--', alpha=0.7, zorder=1)
            
            # Add annotation
            ax.annotate('PPDA: Opponent Passes ÷ Sweden Defensive Actions\nLower PPDA = More intense pressing', 
                      xy=(0.5, 0.02), xycoords='figure fraction', 
                      ha='center', va='bottom', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", fc=SWEDEN_COLORS['background'], 
                              ec=SWEDEN_COLORS['text'], alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(pitch_viz_dir / f"pass_press_map_{match_id}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create average heatmap across all matches
    if all_defensive_x and all_defensive_y:
        # Calculate average PPDA
        avg_ppda = total_ppda / match_count if match_count > 0 else 0
        
        # Create figure for average heatmap
        fig, ax = plt.subplots(figsize=(16, 12))
        pitch.draw(ax=ax)
        
        # Create a heatmap using kernel density estimation
        heatmap, xedges, yedges = np.histogram2d(
            all_defensive_x, all_defensive_y, 
            bins=(25, 25),  # Higher resolution for aggregate view
            range=[[0, 100], [0, 100]]  # Match pitch dimensions
        )
        
        # Smooth the heatmap with a Gaussian filter
        heatmap = ndimage.gaussian_filter(heatmap, sigma=2)
        
        # Normalize the heatmap by the number of matches
        heatmap = heatmap / match_count if match_count > 0 else heatmap
        
        # Create mesh grid for the contour plot with correct dimensions
        x_mesh = np.linspace(0, 100, heatmap.shape[0])
        y_mesh = np.linspace(0, 100, heatmap.shape[1])
        x_grid, y_grid = np.meshgrid(x_mesh, y_mesh)
        
        # Draw the contour plot
        contour = ax.contourf(
            x_grid, y_grid, heatmap.T,  # Transpose for correct orientation
            levels=20,  # More levels for smoother gradient
            cmap='YlOrRd',  # Yellow to Red colormap
            alpha=0.8,
            zorder=1
        )
        
        # Plot Sweden's aggregate defensive actions as scatter points with reduced alpha to avoid overwhelming
        ax.scatter(all_defensive_x, all_defensive_y, c=SWEDEN_COLORS['primary'], 
                  alpha=0.1, s=5, zorder=2, edgecolors=None)
        
        # Add the thirds of the pitch for reference
        ax.axvline(x=33.3, color='white', linestyle='--', alpha=0.7, zorder=2, linewidth=2)
        ax.axvline(x=66.6, color='white', linestyle='--', alpha=0.7, zorder=2, linewidth=2)
            
        # Add text labels for each third
        ax.text(16.5, 5, "Defensive Third", color='white', fontsize=12, ha='center', fontweight='bold')
        ax.text(50, 5, "Middle Third", color='white', fontsize=12, ha='center', fontweight='bold')
        ax.text(83.5, 5, "Attacking Third", color='white', fontsize=12, ha='center', fontweight='bold')
        
        # Add a title and colorbar
        ax.set_title(f"Sweden's Average Pressing Intensity Across All Matches\nAverage PPDA: {avg_ppda:.2f}", 
                   fontsize=20, fontweight='bold')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Defensive Actions Density', fontsize=14)
        
        # Add annotations for the aggregate view
        ax.annotate('Higher intensity areas represent regions where Sweden engages in more defensive actions across all matches', 
                  xy=(0.5, 0.04), xycoords='figure fraction', 
                  ha='center', va='bottom', fontsize=12, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='black', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(pitch_viz_dir / "aggregate_pressing_heatmap.png", dpi=400, bbox_inches='tight')
        plt.close()
        
        # Create a more detailed version with text analysis
        fig, ax = plt.subplots(figsize=(16, 12))
        pitch.draw(ax=ax)
        
        # Draw the same contour plot as above
        contour = ax.contourf(
            x_grid, y_grid, heatmap.T,  
            levels=20,
            cmap='YlOrRd',
            alpha=0.8,
            zorder=1
        )
        
        # Calculate intensity by thirds
        defensive_third_intensity = np.sum(heatmap[0:8, :]) / np.sum(heatmap) * 100
        middle_third_intensity = np.sum(heatmap[8:17, :]) / np.sum(heatmap) * 100
        attacking_third_intensity = np.sum(heatmap[17:, :]) / np.sum(heatmap) * 100
        
        # Add annotations for zone intensity
        plt.annotate(f'Defensive Third: {defensive_third_intensity:.1f}%', 
                   xy=(16.5, 95), xycoords='data', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.8))
                   
        plt.annotate(f'Middle Third: {middle_third_intensity:.1f}%', 
                   xy=(50, 95), xycoords='data', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.8))
                   
        plt.annotate(f'Attacking Third: {attacking_third_intensity:.1f}%', 
                   xy=(83.5, 95), xycoords='data', 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.8))
        
        # Add the thirds of the pitch for reference
        ax.axvline(x=33.3, color='white', linestyle='--', alpha=0.7, zorder=2, linewidth=2)
        ax.axvline(x=66.6, color='white', linestyle='--', alpha=0.7, zorder=2, linewidth=2)
        
        # Add text labels for each third
        ax.text(16.5, 5, "Defensive Third", color='white', fontsize=12, ha='center', fontweight='bold')
        ax.text(50, 5, "Middle Third", color='white', fontsize=12, ha='center', fontweight='bold')
        ax.text(83.5, 5, "Attacking Third", color='white', fontsize=12, ha='center', fontweight='bold')
        
        # Add a title and colorbar
        ax.set_title(f"Sweden's Pressing Distribution Analysis\nAverage PPDA: {avg_ppda:.2f}", 
                   fontsize=20, fontweight='bold')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Defensive Actions Density', fontsize=14)
        
        # Add tactical insight annotation based on the intensity distribution
        if attacking_third_intensity > defensive_third_intensity + 10:
            pressing_style = "High Pressing"
            description = "Sweden predominantly employs a high pressing strategy, engaging opponents in the attacking third"
        elif middle_third_intensity > attacking_third_intensity and middle_third_intensity > defensive_third_intensity:
            pressing_style = "Mid-Block Pressing"
            description = "Sweden primarily focuses on pressing in the middle third, using a mid-block defensive strategy"
        else:
            pressing_style = "Deep Block Defending"
            description = "Sweden tends to defend deep, focusing defensive actions in their own defensive third"
            
        ax.annotate(f'Pressing Style: {pressing_style}\n{description}', 
                  xy=(0.5, 0.04), xycoords='figure fraction', 
                  ha='center', va='bottom', fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='black', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(pitch_viz_dir / "pressing_analysis_summary.png", dpi=400, bbox_inches='tight')
        plt.close()
    
    print(f"Pitch visualizations saved to {pitch_viz_dir}")

def analyze_ppda_vs_outcome(defensive_df):
    """Analyze the relationship between PPDA and match outcomes/goals conceded."""
    print("Analyzing relationship between PPDA and match outcomes...")
    
    # Check if we have outcome data
    if 'outcome' not in defensive_df.columns:
        print("No match outcome data available for analysis.")
        return
    
    # 1. Create a categorical outcome column for visualization
    outcome_colors = {
        'win': SWEDEN_COLORS['secondary'],
        'draw': 'gray',
        'loss': 'red'
    }
    
    outcome_sizes = {
        'win': 150,
        'draw': 100,
        'loss': 80
    }
    
    # 2. PPDA vs Match Outcome
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each outcome category separately to control legend
    for outcome in outcome_colors:
        subset = defensive_df[defensive_df['outcome'] == outcome]
        if not subset.empty:
            ax.scatter(
                subset['ppda'], 
                subset['opponent_goals'],
                label=f"{outcome.capitalize()} ({len(subset)} matches)",
                color=outcome_colors[outcome],
                s=outcome_sizes[outcome],
                alpha=0.7,
                edgecolors='white'
            )
    
    # Add text labels for opponent names and match dates
    for _, row in defensive_df.iterrows():
        if pd.notnull(row.get('outcome')):
            ax.annotate(
                f"{row['opponent']}\n{row['match_date'].strftime('%Y-%m-%d')}",
                xy=(row['ppda'], row['opponent_goals']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.7)
            )
    
    # Customize plot
    ax.set_title('PPDA vs Goals Conceded by Match Outcome', fontsize=16, fontweight='bold')
    ax.set_xlabel('PPDA (lower = more intense pressing)', fontsize=12)
    ax.set_ylabel('Goals Conceded by Sweden', fontsize=12)
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a trend line using linear regression
    if len(defensive_df) > 1:
        try:
            # Simple linear regression
            from scipy import stats
            
            # Filter out missing values
            valid_data = defensive_df.dropna(subset=['ppda', 'opponent_goals'])
            
            if len(valid_data) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data['ppda'], valid_data['opponent_goals']
                )
                
                # Create trend line points
                x_trend = np.linspace(valid_data['ppda'].min(), valid_data['ppda'].max(), 100)
                y_trend = slope * x_trend + intercept
                
                # Plot the trend line
                ax.plot(
                    x_trend, y_trend, 
                    color='black', linestyle='--', 
                    label=f'Trend Line (R² = {r_value**2:.2f})'
                )
                
                # Add annotation about the trend
                trend_description = "Higher PPDA → More Goals Conceded" if slope > 0 else "Lower PPDA → More Goals Conceded"
                ax.annotate(
                    trend_description,
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
                )
        except Exception as e:
            print(f"Error calculating trend line: {e}")
    
    # Add legend
    ax.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppda_vs_goals_conceded.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot of PPDA by match outcome
    if 'outcome' in defensive_df.columns and defensive_df['outcome'].notna().any():
        plt.figure(figsize=(10, 8))
        
        # Create a custom order for the outcomes
        custom_order = ['win', 'draw', 'loss']
        actual_outcomes = [o for o in custom_order if o in defensive_df['outcome'].unique()]
        
        # Create the boxplot
        ax = sns.boxplot(
            x='outcome', 
            y='ppda',
            data=defensive_df,
            order=actual_outcomes,
            palette=[outcome_colors.get(o, 'gray') for o in actual_outcomes]
        )
        
        # Add individual data points for better visualization
        sns.stripplot(
            x='outcome', 
            y='ppda',
            data=defensive_df,
            order=actual_outcomes,
            color='black', 
            size=8, 
            alpha=0.7
        )
        
        # Add sample size annotations
        for i, outcome in enumerate(actual_outcomes):
            count = len(defensive_df[defensive_df['outcome'] == outcome])
            avg_ppda = defensive_df[defensive_df['outcome'] == outcome]['ppda'].mean()
            plt.annotate(
                f'n = {count}\nAvg PPDA: {avg_ppda:.2f}',
                xy=(i, defensive_df['ppda'].min() - 0.1),
                ha='center',
                va='top',
                fontweight='bold',
                fontsize=10
            )
        
        # Customize the plot
        plt.title('PPDA Distribution by Match Outcome', fontsize=16, fontweight='bold')
        plt.xlabel('Match Outcome', fontsize=14)
        plt.ylabel('PPDA (lower = more intense pressing)', fontsize=14)
        
        # Add annotation explaining PPDA
        plt.annotate(
            'Lower PPDA = More intense pressing',
            xy=(0.5, 0.01),
            xycoords='figure fraction',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.7)
        )
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ppda_by_outcome_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. PPDA vs Sweden Goals correlation
    if 'sweden_goals' in defensive_df.columns and defensive_df['sweden_goals'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot
        scatter = ax.scatter(
            defensive_df['ppda'],
            defensive_df['sweden_goals'],
            c=[outcome_colors.get(o, 'gray') for o in defensive_df['outcome']],
            s=100,
            alpha=0.7,
            edgecolors='white'
        )
        
        # Add text labels for opponent names
        for _, row in defensive_df.iterrows():
            if pd.notnull(row.get('sweden_goals')):
                ax.annotate(
                    f"{row['opponent']}\n{row['match_date'].strftime('%Y-%m-%d')}",
                    xy=(row['ppda'], row['sweden_goals']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.7)
                )
        
        # Add a legend for outcomes
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=outcome.capitalize(),
                      markerfacecolor=color, markersize=10)
            for outcome, color in outcome_colors.items()
            if outcome in defensive_df['outcome'].values
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Customize plot
        ax.set_title('Relationship Between PPDA and Goals Scored by Sweden', fontsize=16, fontweight='bold')
        ax.set_xlabel('PPDA (lower = more intense pressing)', fontsize=14)
        ax.set_ylabel('Goals Scored by Sweden', fontsize=14)
        
        # Add a grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a trend line using linear regression
        if len(defensive_df) > 1:
            try:
                # Simple linear regression
                from scipy import stats
                
                # Filter out missing values
                valid_data = defensive_df.dropna(subset=['ppda', 'sweden_goals'])
                
                if len(valid_data) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid_data['ppda'], valid_data['sweden_goals']
                    )
                    
                    # Create trend line points
                    x_trend = np.linspace(valid_data['ppda'].min(), valid_data['ppda'].max(), 100)
                    y_trend = slope * x_trend + intercept
                    
                    # Plot the trend line
                    ax.plot(
                        x_trend, y_trend, 
                        color='black', linestyle='--', 
                        label=f'Trend Line (R² = {r_value**2:.2f})'
                    )
                    
                    # Add trend line details to legend
                    ax.legend(handles=legend_elements + [
                        plt.Line2D([0], [0], color='black', linestyle='--', 
                                 label=f'Trend (R² = {r_value**2:.2f})')
                    ], loc='upper right')
                    
                    # Add annotation about the correlation strength
                    correlation_strength = abs(r_value)
                    if correlation_strength < 0.3:
                        correlation_text = "Weak correlation"
                    elif correlation_strength < 0.7:
                        correlation_text = "Moderate correlation"
                    else:
                        correlation_text = "Strong correlation"
                    
                    direction = "negative" if slope > 0 else "positive"
                    
                    ax.annotate(
                        f"{correlation_text} ({direction})",
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
                    )
            except Exception as e:
                print(f"Error calculating trend line: {e}")
    
    # Add annotation explaining PPDA
    ax.annotate(
        'Lower PPDA = More intense pressing',
        xy=(0.5, 0.01),
        xycoords='figure fraction',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppda_vs_sweden_goals.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("PPDA vs match outcome analysis completed.")

if __name__ == "__main__":
    defensive_df, zones_df, timeline_df = analyze_defensive_metrics()
    
    if defensive_df is not None:
        print("\nDefensive Metrics Summary:")
        print("-" * 50)
        print(f"Total matches analyzed: {len(defensive_df)}")
        print(f"Average PPDA for Sweden: {defensive_df['ppda'].mean():.2f}")
        print(f"Best PPDA: {defensive_df['ppda'].min():.2f} (lower is better)")
        print(f"Worst PPDA: {defensive_df['ppda'].max():.2f}")
        
        # Home vs Away comparison
        home_matches = defensive_df[defensive_df['sweden_status'] == 'Home']
        away_matches = defensive_df[defensive_df['sweden_status'] == 'Away']
        
        if not home_matches.empty:
            print(f"\nHome matches ({len(home_matches)}):")
            print(f"  Average PPDA: {home_matches['ppda'].mean():.2f}")
        
        if not away_matches.empty:
            print(f"\nAway matches ({len(away_matches)}):")
            print(f"  Average PPDA: {away_matches['ppda'].mean():.2f}")
        
        print("\nPPDA by opponent:")
        for opponent, group in defensive_df.groupby('opponent'):
            print(f"  {opponent}: {group['ppda'].mean():.2f}")
        
        print("\nMatch-by-match PPDA:")
        for _, row in defensive_df.iterrows():
            print(f"  {row['match_date']} - {row['home_team']} vs {row['away_team']}: " +
                  f"PPDA = {row['ppda']:.2f}")