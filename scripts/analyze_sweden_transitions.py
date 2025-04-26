#!/usr/bin/env python3
"""
Analyze transition speed after ball recovery for the Swedish women's team.
This script analyzes:
1. Time-to-attack measurements after ball recovery
2. Vertical progression speed analysis 
3. Key transition player identification

The analysis focuses on Sweden's defensive performance when without the ball 
or right after winning it back.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Path to the Sweden data directory
BASE_DIR = Path(__file__).parent.parent
SWEDEN_DATA_DIR = BASE_DIR / "data" / "sweden_data"
WYSCOUT_DIR = SWEDEN_DATA_DIR / "wyscout"
SKILLCORNER_DIR = SWEDEN_DATA_DIR / "skillcorner"
OUTPUT_DIR = SWEDEN_DATA_DIR / "analysis" / "defensive_transitions"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sweden team colors
SWEDEN_BLUE = '#0072C6'
SWEDEN_YELLOW = '#FFDA00'

# Define transition window time (in seconds)
TRANSITION_WINDOW = 10  # Analyze first 10 seconds after ball recovery

def load_wyscout_data(file_path):
    """Load Wyscout event data from a JSON file."""
    with open(file_path, encoding='utf-8') as f:
        js = json.load(f)
        df = pd.json_normalize(js['events'])
    return df

def find_ball_recovery_events(events_df):
    """
    Find all ball recovery events for Sweden in the Wyscout data.
    This includes interceptions, tackles, and other defensive actions that
    result in winning the ball.
    """
    # Filter for Swedish team (adjust team ID as needed)
    sweden_team_id = identify_sweden_team_id(events_df)
    
    if sweden_team_id is None:
        print("Could not identify Sweden team ID")
        return pd.DataFrame()
    
    print(f"Identified Sweden team ID: {sweden_team_id}")
    
    # Filter for ball recovery events - expanded to catch more recovery events
    recovery_events = events_df[
        (events_df['team.id'] == sweden_team_id) & 
        ((events_df['type.primary'] == 'interception') | 
         (events_df['type.primary'] == 'duel') |
         (events_df['type.primary'] == 'recovery') |
         (events_df['type.secondary'] == 'ball_recovery') |
         (events_df['type.primary'] == 'tackle'))
    ].copy()
    
    print(f"Found {len(recovery_events)} potential ball recovery events")
    
    # Extract location information - with error handling
    recovery_events['recovery_x'] = recovery_events.apply(
        lambda row: row.get('location.x', None) if 'location.x' in row else 
                  (row['location'][0] if isinstance(row.get('location', None), list) and len(row['location']) > 0 else 50), 
        axis=1
    )
    
    recovery_events['recovery_y'] = recovery_events.apply(
        lambda row: row.get('location.y', None) if 'location.y' in row else 
                  (row['location'][1] if isinstance(row.get('location', None), list) and len(row['location']) > 1 else 50), 
        axis=1
    )
    
    # Extract timestamp
    recovery_events['timestamp'] = recovery_events['matchTimestamp']
    
    # Extract player information with error handling
    recovery_events['player_name'] = recovery_events.apply(
        lambda row: row.get('player.name', 'Unknown Player'),
        axis=1
    )
    
    recovery_events['player_id'] = recovery_events.apply(
        lambda row: row.get('player.id', 0),
        axis=1
    )
    
    return recovery_events

def identify_sweden_team_id(events_df):
    """Identify Sweden's team ID from the events data."""
    # Look for "Sweden" in team names
    teams = events_df['team.name'].unique()
    for team in teams:
        if isinstance(team, str) and "Sweden" in team:
            sweden_id = events_df[events_df['team.name'] == team]['team.id'].iloc[0]
            return sweden_id
    
    # If not found by name, try to infer from the filename or other contextual clues
    # This is a fallback method
    if 'team.id' in events_df.columns and not events_df['team.id'].empty:
        potential_ids = events_df['team.id'].value_counts().index
        if len(potential_ids) >= 2:
            # If we have at least two team IDs, return the second most common
            # (assuming home team might have more events in some matches)
            return potential_ids[1]
        elif len(potential_ids) == 1:
            return potential_ids[0]
    
    print("WARNING: Could not identify Sweden team ID in the match data")
    return None

def convert_timestamp_to_milliseconds(timestamp_str):
    """
    Convert a timestamp string in format 'HH:MM:SS.mmm' to milliseconds.
    """
    try:
        # Split into time part and milliseconds
        time_parts, milliseconds = timestamp_str.split('.')
        hours, minutes, seconds = map(int, time_parts.split(':'))
        
        # Convert to total milliseconds
        total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + int(milliseconds)
        return total_ms
    except Exception as e:
        print(f"Error converting timestamp '{timestamp_str}': {str(e)}")
        return 0  # Default value if conversion fails

def analyze_transitions(match_file, tracking_dir=None):
    """
    Analyze transitions for a single match.
    Returns transition metrics for the match.
    """
    # Load Wyscout event data
    try:
        events_df = load_wyscout_data(match_file)
        print(f"Loaded {len(events_df)} events from {match_file.name}")
        
        # Print available columns to help with debugging
        #print(f"Available columns: {', '.join(events_df.columns)}")
        
        # Print team IDs and names for better understanding
        if 'team.id' in events_df.columns and 'team.name' in events_df.columns:
            team_info = events_df[['team.id', 'team.name']].drop_duplicates()
            print(f"Teams in match: {team_info.to_dict('records')}")
            
        # Convert matchTimestamp from time format to milliseconds
        if 'matchTimestamp' in events_df.columns:
            if events_df['matchTimestamp'].dtype == 'object':
                print("Converting matchTimestamp from time format to milliseconds")
                events_df['matchTimestamp'] = events_df['matchTimestamp'].apply(convert_timestamp_to_milliseconds)
    except Exception as e:
        print(f"Error loading match file {match_file}: {str(e)}")
        return pd.DataFrame()
    
    # Extract match information
    match_id = match_file.stem.split('_')[0]
    match_date = match_file.stem.split('_')[-1]
    
    # Find all ball recovery events
    recoveries = find_ball_recovery_events(events_df)
    
    # If we have no recoveries, return empty results
    if len(recoveries) == 0:
        print(f"No ball recoveries found for match {match_id}")
        return pd.DataFrame()
    
    # Find subsequent events after each recovery
    transition_metrics = []
    
    for _, recovery in recoveries.iterrows():
        recovery_time = recovery['timestamp']
        recovery_x = recovery['recovery_x']
        recovery_y = recovery['recovery_y']
        player_name = recovery['player_name']
        player_id = recovery['player_id']
        
        # Ensure recovery_time is numeric
        if isinstance(recovery_time, str):
            recovery_time = convert_timestamp_to_milliseconds(recovery_time)
        
        # Find all events within transition window after recovery
        transition_events = events_df[
            (events_df['matchTimestamp'] > recovery_time) & 
            (events_df['matchTimestamp'] <= recovery_time + TRANSITION_WINDOW * 1000) &
            (events_df['team.id'] == recovery['team.id'])
        ].sort_values('matchTimestamp')
        
        if len(transition_events) == 0:
            continue
        
        # Calculate time to first forward pass or shot
        attacking_events = transition_events[
            (transition_events['type.primary'].isin(['pass', 'shot', 'cross', 'free_kick', 'corner_kick', 'penalty']))
        ]
        
        if len(attacking_events) > 0:
            first_attack = attacking_events.iloc[0]
            time_to_attack = (first_attack['matchTimestamp'] - recovery_time) / 1000  # in seconds
            
            # Calculate vertical progression
            end_x = None
            
            # Try different ways to get the location data
            if 'location.x' in first_attack and pd.notna(first_attack['location.x']):
                end_x = first_attack['location.x']
            elif 'location' in first_attack and isinstance(first_attack['location'], list) and len(first_attack['location']) > 0:
                end_x = first_attack['location'][0]
            elif 'pass.endLocation.x' in first_attack and pd.notna(first_attack['pass.endLocation.x']):
                end_x = first_attack['pass.endLocation.x']
            
            # If we have location data, calculate progression
            if end_x is not None and recovery_x is not None:
                try:
                    # Make sure values are numeric
                    end_x = float(end_x)
                    recovery_x = float(recovery_x)
                    vert_progression = end_x - recovery_x
                    
                    # Calculate progression speed (field units per second)
                    progression_speed = vert_progression / time_to_attack if time_to_attack > 0 else 0
                    
                    # Store metrics
                    transition_metrics.append({
                        'match_id': match_id,
                        'match_date': match_date,
                        'recovery_time': recovery_time,
                        'recovery_x': recovery_x,
                        'recovery_y': recovery_y,
                        'recovery_player': player_name,
                        'recovery_player_id': player_id,
                        'time_to_attack': time_to_attack,
                        'vert_progression': vert_progression,
                        'progression_speed': progression_speed,
                        'first_action_type': first_attack['type.primary'],
                        'action_player': first_attack.get('player.name', 'Unknown')
                    })
                except (ValueError, TypeError) as e:
                    print(f"Error calculating progression: {str(e)}")
    
    if not transition_metrics:
        print(f"No valid transitions found for match {match_id}")
    else:
        print(f"Found {len(transition_metrics)} valid transitions for match {match_id}")
    
    return pd.DataFrame(transition_metrics) if transition_metrics else pd.DataFrame()

def analyze_all_matches():
    """Analyze transitions for all Sweden matches."""
    # Find all match files containing Sweden
    match_files = list(WYSCOUT_DIR.glob("*Sweden*.json"))
    
    if not match_files:
        print(f"No Sweden match files found in {WYSCOUT_DIR}")
        # Try an alternative broader search if specific pattern fails
        match_files = list(WYSCOUT_DIR.glob("*.json"))
        if match_files:
            print(f"Found {len(match_files)} general match files instead")
        else:
            return None
    
    print(f"Found {len(match_files)} Sweden match files")
    
    all_transitions = []
    
    for match_file in match_files:
        print(f"Analyzing transitions for match: {match_file.name}")
        transitions = analyze_transitions(match_file)
        if not transitions.empty:
            all_transitions.append(transitions)
            print(f"Successfully extracted {len(transitions)} transitions")
        else:
            print(f"No transitions extracted from {match_file.name}")
    
    if not all_transitions:
        print("No transition data found across matches")
        return None
    
    # Combine data from all matches
    combined_transitions = pd.concat(all_transitions, ignore_index=True)
    print(f"Combined {len(combined_transitions)} transitions from all matches")
    
    # Save the data
    combined_transitions.to_csv(OUTPUT_DIR / "transition_metrics.csv", index=False)
    
    return combined_transitions

def visualize_time_to_attack(transition_data):
    """
    Visualize the time-to-attack distribution after ball recovery.
    """
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create histogram
    ax = sns.histplot(
        data=transition_data, 
        x="time_to_attack", 
        bins=20, 
        kde=True, 
        color=SWEDEN_BLUE
    )
    
    # Add vertical line for mean and median
    mean_time = transition_data['time_to_attack'].mean()
    median_time = transition_data['time_to_attack'].median()
    
    plt.axvline(mean_time, color=SWEDEN_YELLOW, linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f}s')
    plt.axvline(median_time, color='red', linestyle=':', linewidth=2, label=f'Median: {median_time:.2f}s')
    
    # Add labels and title
    plt.xlabel('Time to First Attacking Action (seconds)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Sweden: Time to First Attacking Action After Ball Recovery', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    
    # Limit x-axis to transition window
    plt.xlim(0, TRANSITION_WINDOW)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_to_attack_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_recovery_locations(transition_data):
    """
    Create a heatmap of ball recovery locations.
    """
    plt.figure(figsize=(12, 8))
    
    # Create a pitch
    pitch = create_pitch()
    
    # Create a heatmap of recovery locations
    hb = plt.hexbin(
        transition_data['recovery_x'], 
        transition_data['recovery_y'], 
        gridsize=20, 
        cmap='YlOrRd', 
        alpha=0.7,
        extent=[0, 100, 0, 100]  # Wyscout uses 0-100 coordinates
    )
    
    # Add a color bar
    cb = plt.colorbar(hb, ax=pitch)
    cb.set_label('Number of Ball Recoveries', fontsize=12)
    
    # Add title and labels
    plt.title('Sweden: Ball Recovery Locations', fontsize=16, fontweight='bold')
    plt.xlabel('Length of Pitch (Attack →)', fontsize=14)
    plt.ylabel('Width of Pitch', fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "recovery_locations_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_pitch():
    """Create a football pitch for visualizations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pitch outline
    plt.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color='black')
    
    # Halfway line
    plt.plot([50, 50], [0, 100], color='black')
    
    # Center circle
    center_circle = plt.Circle((50, 50), 9.15, fill=False, color='black')
    ax.add_patch(center_circle)
    
    # Penalty areas
    plt.plot([0, 16.5, 16.5, 0], [21.1, 21.1, 78.9, 78.9], color='black')
    plt.plot([100, 83.5, 83.5, 100], [21.1, 21.1, 78.9, 78.9], color='black')
    
    # Goal areas
    plt.plot([0, 5.5, 5.5, 0], [36.8, 36.8, 63.2, 63.2], color='black')
    plt.plot([100, 94.5, 94.5, 100], [36.8, 36.8, 63.2, 63.2], color='black')
    
    # Goals
    plt.plot([-2, 0], [45.2, 45.2], color='black')
    plt.plot([-2, 0], [54.8, 54.8], color='black')
    plt.plot([-2, -2], [45.2, 54.8], color='black')
    
    plt.plot([100, 102], [45.2, 45.2], color='black')
    plt.plot([100, 102], [54.8, 54.8], color='black')
    plt.plot([102, 102], [45.2, 54.8], color='black')
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Set limits
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    
    return ax

def visualize_progression_speed(transition_data):
    """
    Visualize the vertical progression speed after ball recovery.
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data - filter out negative progression (backward passes)
    positive_progression = transition_data[transition_data['vert_progression'] > 0].copy()
    
    # If we don't have any positive progression data, return
    if len(positive_progression) == 0:
        print("No positive progression data found")
        return
    
    # Create a custom colormap for Sweden's colors
    sweden_cmap = LinearSegmentedColormap.from_list('sweden', [SWEDEN_BLUE, SWEDEN_YELLOW])
    
    # Create scatter plot
    scatter = plt.scatter(
        positive_progression['recovery_x'], 
        positive_progression['progression_speed'],
        c=positive_progression['vert_progression'], 
        cmap=sweden_cmap,
        alpha=0.7,
        s=100,
        edgecolor='black'
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Vertical Progression (field units)', fontsize=12)
    
    # Add labels and title
    plt.xlabel('Recovery X-Position (Attack →)', fontsize=14)
    plt.ylabel('Progression Speed (field units/second)', fontsize=14)
    plt.title('Sweden: Vertical Progression Speed vs Recovery Position', fontsize=16, fontweight='bold')
    
    # Add trend line
    x = positive_progression['recovery_x']
    y = positive_progression['progression_speed']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.7, linewidth=2)
    
    # Add correlation coefficient
    corr = np.corrcoef(x, y)[0, 1]
    plt.annotate(f"Correlation: {corr:.2f}", xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "progression_speed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def identify_key_transition_players(transition_data):
    """
    Identify and visualize key players in transitions.
    """
    plt.figure(figsize=(14, 10))
    
    # Group data by recovery player
    player_metrics = transition_data.groupby('recovery_player').agg({
        'recovery_player_id': 'first',  # Just get the ID
        'time_to_attack': ['mean', 'count'],
        'vert_progression': ['mean', 'sum'],
        'progression_speed': 'mean'
    }).reset_index()
    
    # Flatten the column names
    player_metrics.columns = ['_'.join(col).strip('_') for col in player_metrics.columns.values]
    
    # Filter for players with multiple recoveries (more reliable data)
    min_recoveries = 3  # Minimum number of recoveries to be included
    key_players = player_metrics[player_metrics['time_to_attack_count'] >= min_recoveries].copy()
    
    # If we don't have enough data, return
    if len(key_players) == 0:
        print("Not enough data to identify key transition players")
        return
    
    # Sort by various metrics
    key_players_by_time = key_players.sort_values('time_to_attack_mean')
    key_players_by_progression = key_players.sort_values('vert_progression_sum', ascending=False)
    key_players_by_speed = key_players.sort_values('progression_speed_mean', ascending=False)
    
    # Create scatter plot: Time to Attack vs Progression Speed
    scatter = plt.scatter(
        key_players['time_to_attack_mean'],
        key_players['progression_speed_mean'],
        s=key_players['time_to_attack_count'] * 20,  # Size based on number of recoveries
        c=key_players['vert_progression_mean'],  # Color based on average progression
        cmap='viridis',
        alpha=0.7,
        edgecolor='black'
    )
    
    # Add player labels for top performers
    for idx, player in key_players.iterrows():
        # Label top players by either metric
        if (player['time_to_attack_mean'] < key_players['time_to_attack_mean'].median() or 
            player['progression_speed_mean'] > key_players['progression_speed_mean'].median()):
            plt.annotate(
                player['recovery_player'],
                (player['time_to_attack_mean'], player['progression_speed_mean']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
    
    # Add a legend for bubble size
    sizes = key_players['time_to_attack_count'].unique()
    size_legend_elements = []
    for size in sorted(sizes):
        size_legend_elements.append(
            mpatches.Circle((0, 0), radius=np.sqrt(size * 20 / np.pi), 
                          label=f'{size} recoveries', alpha=0.5, color=SWEDEN_BLUE)
        )
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Vertical Progression (field units)', fontsize=12)
    
    # Add second legend for bubble size
    
    # Create a custom legend for bubble sizes
    sizes = sorted(key_players['time_to_attack_count'].unique())
    handles = []
    for size in sizes:
        handles.append(plt.scatter([], [], s=size*20, color=SWEDEN_BLUE, alpha=0.5, 
                                 label=f'{size} recoveries', edgecolor='black'))
    
    # Slice the handles to display only a few bubbles
    handles_to_show = handles[::5][:-1:] # find a way to append(handles[len(handles)-1])
    
    # Add second legend for bubble size
    plt.legend(handles=handles_to_show, title="Number of Recoveries", 
              loc='upper right', fontsize=10, scatterpoints=1, 
              labelspacing=2.5, borderpad=1, handletextpad=2)
    
    # Add labels and title
    plt.xlabel('Average Time to First Attacking Action (s)', fontsize=14)
    plt.ylabel('Average Progression Speed (field units/s)', fontsize=14)
    plt.title('Sweden: Key Players in Defensive Transitions', fontsize=16, fontweight='bold')
    
    # Add quadrant lines (median values)
    plt.axvline(x=key_players['time_to_attack_mean'].median(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=key_players['progression_speed_mean'].median(), color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    plt.text(
        0.25, 0.75, 
        "FAST TRANSITIONS\nHIGH PROGRESSION", 
        transform=plt.gca().transAxes, 
        fontsize=12, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.5)
    )
    plt.text(
        0.7, 0.75, 
        "SLOW TRANSITIONS\nHIGH PROGRESSION", 
        transform=plt.gca().transAxes, 
        fontsize=12, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.5)
    )
    plt.text(
        0.25, 0.25, 
        "FAST TRANSITIONS\nLOW PROGRESSION", 
        transform=plt.gca().transAxes, 
        fontsize=12, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.5)
    )
    plt.text(
        0.75, 0.25, 
        "SLOW TRANSITIONS\nLOW PROGRESSION", 
        transform=plt.gca().transAxes, 
        fontsize=12, ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.5)
    )
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "key_transition_players.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a bar chart of top players by total vertical progression contribution
    plt.figure(figsize=(14, 8))
    
    # Get top 10 players by progression
    top_players = key_players_by_progression.head(10)
    
    # Create bar chart
    bars = plt.bar(
        top_players['recovery_player'],
        top_players['vert_progression_sum'],
        color=SWEDEN_BLUE,
        edgecolor='black',
        alpha=0.7
    )
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=10
        )
    
    # Add labels and title
    plt.xlabel('Player', fontsize=14)
    plt.ylabel('Total Vertical Progression (field units)', fontsize=14)
    plt.title('Sweden: Top Players by Total Vertical Progression Contribution', fontsize=16, fontweight='bold')
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_progression_players.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save key player data
    key_players.to_csv(OUTPUT_DIR / "key_transition_players.csv", index=False)
    
    return key_players

def create_transition_zone_analysis(transition_data):
    """
    Analyze transitions based on where the ball was recovered.
    Divide the pitch into zones and analyze transition metrics by zone.
    """
    # Create pitch zones (3 horizontal x 3 vertical = 9 zones)
    transition_data['zone_x'] = pd.cut(
        transition_data['recovery_x'], 
        bins=[0, 33.3, 66.6, 100], 
        labels=['Defensive Third', 'Middle Third', 'Attacking Third']
    )
    transition_data['zone_y'] = pd.cut(
        transition_data['recovery_y'], 
        bins=[0, 33.3, 66.6, 100], 
        labels=['Left', 'Center', 'Right']
    )
    
    # Combine to create zone name
    transition_data['zone'] = transition_data['zone_x'].astype(str) + ' - ' + transition_data['zone_y'].astype(str)
    
    # Group by zone
    zone_metrics = transition_data.groupby('zone').agg({
        'time_to_attack': ['mean', 'count'],
        'vert_progression': 'mean',
        'progression_speed': 'mean'
    }).reset_index()
    
    # Flatten the column names
    zone_metrics.columns = ['_'.join(col).strip('_') for col in zone_metrics.columns.values]
    
    # Reshape the data for the heatmap
    # First, split the zone back into x and y components
    zone_metrics[['zone_x', 'zone_y']] = zone_metrics['zone'].str.split(' - ', expand=True)
    
    # Create a field visualization for time to attack
    plt.figure(figsize=(14, 8))
    
    # Create a pitch
    pitch = create_pitch()
    
    # Create a pivot table for time to attack
    time_heatmap_data = zone_metrics.pivot_table(
        index='zone_x', 
        columns='zone_y', 
        values='time_to_attack_mean'
    )
    
    # Print debug information
    print("Zone metrics data (time to attack):")
    print(zone_metrics.head())
    print("\nPivot table data (time to attack):")
    print(time_heatmap_data)
    
    # Define the zone coordinates on the pitch (center of each zone)
    zone_coordinates = {
        ('Defensive Third', 'Left'): (16.65, 16.65),
        ('Defensive Third', 'Center'): (16.65, 50.0),
        ('Defensive Third', 'Right'): (16.65, 83.35),
        ('Middle Third', 'Left'): (50.0, 16.65),
        ('Middle Third', 'Center'): (50.0, 50.0),
        ('Middle Third', 'Right'): (50.0, 83.35),
        ('Attacking Third', 'Left'): (83.35, 16.65),
        ('Attacking Third', 'Center'): (83.35, 50.0),
        ('Attacking Third', 'Right'): (83.35, 83.35)
    }
    
    # Size of each zone rectangle
    zone_width = 33.3
    zone_height = 33.3
    
    # Get min and max values for color scaling
    vmin = time_heatmap_data.min().min()
    vmax = time_heatmap_data.max().max()
    
    # Create a colormap - Yellow-Orange-Red reversed (yellow = faster = better)
    cmap = plt.cm.YlOrRd_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Draw each zone as a rectangle with color based on time to attack
    for x_zone in ['Defensive Third', 'Middle Third', 'Attacking Third']:
        for y_zone in ['Left', 'Center', 'Right']:
            try:
                # Get the value for this zone - handle missing data gracefully
                if x_zone in time_heatmap_data.index and y_zone in time_heatmap_data.columns:
                    value = time_heatmap_data.loc[x_zone, y_zone]
                    
                    # Skip NaN values
                    if pd.isna(value):
                        print(f"Skipping {x_zone}-{y_zone} (NaN value)")
                        continue
                    
                    # Calculate rectangle coordinates
                    center_x, center_y = zone_coordinates[(x_zone, y_zone)]
                    rect_x = center_x - zone_width/2
                    rect_y = center_y - zone_height/2
                    
                    # Create rectangle patch colored by time to attack
                    rect = plt.Rectangle(
                        (rect_x, rect_y), 
                        zone_width, 
                        zone_height, 
                        facecolor=cmap(norm(value)), 
                        alpha=0.7,
                        edgecolor='black',
                        zorder=2
                    )
                    pitch.add_patch(rect)
                    
                    # Add text with value
                    plt.text(
                        center_x, 
                        center_y, 
                        f'{value:.2f}s', 
                        ha='center', 
                        va='center', 
                        fontsize=12, 
                        fontweight='bold',
                        color='black',
                        zorder=3
                    )
                else:
                    print(f"Zone {x_zone}-{y_zone} not found in pivot table")
            except Exception as e:
                print(f"Error processing zone {x_zone}-{y_zone}: {str(e)}")
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=pitch)
    cbar.set_label('Avg. Time to Attack (s)', fontsize=12)
    
    # Add title
    plt.title('Sweden: Average Time to Attack by Pitch Zone', fontsize=16, fontweight='bold')
    
    # Add zone labels for clarity
    label_x = {'Defensive Third': 16.65, 'Middle Third': 50.0, 'Attacking Third': 83.35}
    for x_label, x_pos in label_x.items():
        plt.text(x_pos, -2, x_label, ha='center', va='top', fontsize=10, fontweight='bold')
    
    label_y = {'Left': 16.65, 'Center': 50.0, 'Right': 83.35}
    for y_label, y_pos in label_y.items():
        plt.text(-2, y_pos, y_label, ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    
    # Add arrows to indicate attack direction
    plt.arrow(25, -4, 50, 0, head_width=2, head_length=5, fc='black', ec='black')
    plt.text(50, -7, 'Attack Direction', ha='center', va='center', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transition_zone_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a field visualization for progression speed
    plt.figure(figsize=(14, 8))
    
    # Create a pitch
    pitch = create_pitch()
    
    # Create a pivot table for progression speed
    speed_heatmap_data = zone_metrics.pivot_table(
        index='zone_x', 
        columns='zone_y', 
        values='progression_speed_mean'
    )
    
    # Print debug information
    print("Zone metrics data:")
    print(zone_metrics.head())
    print("\nPivot table data:")
    print(speed_heatmap_data)
    
    # Define the zone coordinates on the pitch (center of each zone)
    zone_coordinates = {
        ('Defensive Third', 'Left'): (16.65, 16.65),
        ('Defensive Third', 'Center'): (16.65, 50.0),
        ('Defensive Third', 'Right'): (16.65, 83.35),
        ('Middle Third', 'Left'): (50.0, 16.65),
        ('Middle Third', 'Center'): (50.0, 50.0),
        ('Middle Third', 'Right'): (50.0, 83.35),
        ('Attacking Third', 'Left'): (83.35, 16.65),
        ('Attacking Third', 'Center'): (83.35, 50.0),
        ('Attacking Third', 'Right'): (83.35, 83.35)
    }
    
    # Size of each zone rectangle
    zone_width = 33.3
    zone_height = 33.3
    
    # Get min and max values for color scaling
    vmin = speed_heatmap_data.min().min()
    vmax = speed_heatmap_data.max().max()
    
    # Create a colormap - Yellow-Green-Blue works well
    cmap = plt.cm.YlGnBu
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Draw each zone as a rectangle with color based on progression speed
    for x_zone in ['Defensive Third', 'Middle Third', 'Attacking Third']:
        for y_zone in ['Left', 'Center', 'Right']:
            try:
                # Get the value for this zone - handle missing data gracefully
                if x_zone in speed_heatmap_data.index and y_zone in speed_heatmap_data.columns:
                    value = speed_heatmap_data.loc[x_zone, y_zone]
                    
                    # Skip NaN values
                    if pd.isna(value):
                        print(f"Skipping {x_zone}-{y_zone} (NaN value)")
                        continue
                    
                    # Calculate rectangle coordinates
                    center_x, center_y = zone_coordinates[(x_zone, y_zone)]
                    rect_x = center_x - zone_width/2
                    rect_y = center_y - zone_height/2
                    
                    # Create rectangle patch colored by progression speed
                    rect = plt.Rectangle(
                        (rect_x, rect_y), 
                        zone_width, 
                        zone_height, 
                        facecolor=cmap(norm(value)), 
                        alpha=0.7,
                        edgecolor='black',
                        zorder=2
                    )
                    pitch.add_patch(rect)
                    
                    # Add text with value
                    plt.text(
                        center_x, 
                        center_y, 
                        f'{value:.2f}', 
                        ha='center', 
                        va='center', 
                        fontsize=12, 
                        fontweight='bold',
                        color='black',
                        zorder=3
                    )
                else:
                    print(f"Zone {x_zone}-{y_zone} not found in pivot table")
            except Exception as e:
                print(f"Error processing zone {x_zone}-{y_zone}: {str(e)}")
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=pitch)
    cbar.set_label('Avg. Progression Speed (field units/s)', fontsize=12)
    
    # Add title
    plt.title('Sweden: Average Progression Speed by Pitch Zone', fontsize=16, fontweight='bold')
    
    # Add zone labels for clarity
    label_x = {'Defensive Third': 16.65, 'Middle Third': 50.0, 'Attacking Third': 83.35}
    for x_label, x_pos in label_x.items():
        plt.text(x_pos, -2, x_label, ha='center', va='top', fontsize=10, fontweight='bold')
    
    label_y = {'Left': 16.65, 'Center': 50.0, 'Right': 83.35}
    for y_label, y_pos in label_y.items():
        plt.text(-2, y_pos, y_label, ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)
    
    # Add arrows to indicate attack direction
    plt.arrow(25, -4, 50, 0, head_width=2, head_length=5, fc='black', ec='black')
    plt.text(50, -7, 'Attack Direction', ha='center', va='center', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "progression_speed_by_zone.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save zone metrics
    zone_metrics.to_csv(OUTPUT_DIR / "transition_zone_metrics.csv", index=False)
    
    return zone_metrics

def main():
    # Analyze transitions for all matches
    print("Analyzing defensive transitions after ball recovery...")
    transition_data = analyze_all_matches()
    
    if transition_data is None or len(transition_data) == 0:
        print("No transition data found. Please check your data files.")
        return
    
    print(f"Found {len(transition_data)} transitions across all matches.")
    
    # Create visualizations
    print("Creating time-to-attack distribution visualization...")
    visualize_time_to_attack(transition_data)
    
    print("Creating ball recovery locations heatmap...")
    visualize_recovery_locations(transition_data)
    
    print("Analyzing vertical progression speed...")
    visualize_progression_speed(transition_data)
    
    print("Identifying key transition players...")
    identify_key_transition_players(transition_data)
    
    print("Creating transition zone analysis...")
    create_transition_zone_analysis(transition_data)
    
    print(f"Analysis complete. Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()