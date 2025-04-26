# Swedish Women's Team Defensive Analysis

## Project Overview
This repository contains our analysis of the Swedish women's national team's defensive behaviors and transitions for the Soccer Analytics course (FS 2025). The analysis focuses on their behaviors when out of possession and during transitions after winning the ball, ahead of the Women's Euro.

## Data Sources
- **Wyscout Event Data**: JSON files containing approximately 1500-2000 events per match
- **SkillCorner Tracking Data**: Comprehensive tracking data with match information and derived data on passes, runs, pressures, and physical performance
- **Video Clips**: Generated using the provided `eventvideo.py` script to illustrate specific defensive scenarios

## Analysis Framework

### 1. Team Defensive Structure & Organization

#### Pressing Analysis
- PPDA (Passes Allowed Per Defensive Action) across matches
- Pressing triggers identification
- Zonal pressing intensity mapping
- Pressing effectiveness metrics (ball recoveries after pressing events)

#### Defensive Shape
- Formation analysis using tracking data
- Compactness measurements (distance between defensive lines)
- Defensive line height variation
- Team width in different defensive phases

### 2. Individual & Unit Behaviors

#### Player-Specific Defensive Roles
- Individual defensive responsibility mapping
- Player-specific metrics (tackles, interceptions, recoveries)
- Positional heat maps during defensive phases

#### Defensive Units
- Inter-unit coordination analysis
- Intra-unit spacing and movement patterns
- Partnership effectiveness metrics

### 3. Ball Recovery & Transition Analysis

#### Ball Recovery Patterns
- Recovery location mapping
- Recovery type categorization
- Zone-specific recovery success rates

#### Transition Speed
- Time-to-attack measurements after ball recovery
- Vertical progression speed analysis
- Key transition player identification

### 4. Defensive Scenarios Analysis

#### Set Piece Defense
- Set piece defensive setup analysis
- Effectiveness against different delivery types
- Individual marking/zonal responsibilities

#### Counter-Press After Loss
- Reaction time measurements
- Immediate recovery success rate
- Counter-pressing intensity by zone

### 5. Opposition-Specific Adaptations
- Comparative defensive approach analysis
- Tactical adjustments based on opposition
- Performance differences against varied opponents

## Implementation Details

### Data Processing Scripts
- `process_event_data.py`: Extract defensive events from Wyscout data
- `analyze_tracking.py`: Process SkillCorner tracking data for defensive positioning
- `transition_analyzer.py`: Identify and analyze transition moments

### Visualization Tools
- Defensive action heat maps
- Network analysis diagrams
- Time-series performance charts
- Video clip compilation of key defensive scenarios

### Metrics Library
- Custom defensive performance indicators
- Team and individual defensive metrics
- Transition effectiveness measurements

## Report Structure

1. **Executive Summary**: Key insights and recommendations
2. **Introduction & Methodology**: Analysis approach and metrics focus
3. **Team Defensive Organization**: Style, structure, pressing patterns
4. **Defensive Transitions**: Recovery analysis and transition effectiveness
5. **Player & Unit Analysis**: Key performers and unit coordination
6. **Opposition-Specific Analysis**: Tactical variations across matches
7. **Strengths & Vulnerabilities**: Identified strengths and areas for improvement
8. **Euro Preparation Recommendations**: Tactical and training suggestions

## Usage Guidelines

1. Clone this repository
2. Set up the environment using `requirements.txt`
3. Place Wyscout and SkillCorner data in the appropriate directories
4. Run analysis scripts in the suggested order:
   - First: Data preprocessing
   - Second: Core defensive analysis
   - Third: Transition analysis
   - Fourth: Visualization generation

## Ethical Considerations
Remember that we have formally agreed not to share the provided data beyond this course. All analysis and findings should remain within the course context.

## Contributors
- [Team Member Names]
- Soccer Analytics Course (FS 2025)
