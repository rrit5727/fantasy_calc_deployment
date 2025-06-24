import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from itertools import combinations
from datetime import datetime
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

@dataclass
class Player:
    name: str
    price: int
    position: str
    secondary_position: str = None
    team: str = None
    projection: float = 0
    diff: float = 0


def load_data() -> pd.DataFrame:
    """
    Load data from PostgreSQL database and rename columns to match expected names.
    
    Returns:
    pd.DataFrame: DataFrame with standardized column names
    """
    # Read database connection parameters from environment
    load_dotenv()
    
    # Get the database URL from Heroku environment or local env file
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        # Handle Heroku's postgres:// URL format
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        # Create SQLAlchemy engine
        engine = create_engine(database_url)
    else:
        # Use individual connection parameters if DATABASE_URL is not available
        db_params = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_DATABASE'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT')
        }
        
        # Create connection string
        conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
        engine = create_engine(conn_str)
    
    try:
        # See what columns are actually in the database
        with engine.connect() as connection:
            # Get column names from the table
            query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'player_stats';
            """
            db_columns = pd.read_sql(query, connection)
            
            # Now fetch the actual data
            query = "SELECT * FROM player_stats;"
            df = pd.read_sql(query, connection)
        
        # Ensure required columns exist
        required_columns = ['Round', 'Team', 'POS1', 'Player', 'Price', 'Diff', 'Projection']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert Round to integer
        df['Round'] = df['Round'].astype(int)
        
        # Handle POS2 if it exists
        if 'POS2' not in df.columns:
            df['POS2'] = None
            
        return df
        
    except Exception as e:
        print(f"Error loading data from database: {str(e)}")
        raise


def get_rounds_data(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split consolidated data into list of DataFrames by round.
    """
    rounds = sorted(df['Round'].unique())
    return [df[df['Round'] == round_num].copy() for round_num in rounds]


def get_traded_out_positions(traded_out_players: List[str], consolidated_data: pd.DataFrame) -> List[str]:
    """
    Get the positions of the players being traded out.
    
    Parameters:
    traded_out_players (List[str]): List of player names being traded out
    consolidated_data (pd.DataFrame): Full dataset containing player information
    
    Returns:
    List[str]: List of positions corresponding to traded out players
    """
    positions = []
    for player in traded_out_players:
        player_data = consolidated_data[consolidated_data['Player'] == player].sort_values('Round', ascending=False)
        if not player_data.empty:
            positions.append(player_data.iloc[0]['POS1'])
    return positions


def get_locked_out_players(simulate_datetime: str, consolidated_data: pd.DataFrame) -> set:
    """
    Get the set of players who are locked out based on the simulated date/time.
    """
    if not simulate_datetime:
        return set()
    
    # Parse the simulated date/time
    simulate_dt = datetime.strptime(simulate_datetime, '%Y-%m-%dT%H:%M')
    
    # Define the fixtures
    fixtures = [
        ("2025-06-26 19:50", ["PEN", "CBY"]),  # Panthers vs Bulldogs
        ("2025-06-27 18:00", ["MAN", "WST"]),  # Sea Eagles vs Tigers
        ("2025-06-27 20:00", ["NEW", "CAN"]),  # Knights vs Raiders
        ("2025-06-28 15:00", ["BRI", "WAR"]),  # Broncos vs Warriors
        ("2025-06-28 17:30", ["SGI", "PAR"]),  # Dragons vs Eels
        ("2025-06-28 19:50", ["DOL", "SOU"]),  # Dolphins vs Rabbitohs
        ("2025-06-29 14:00", ["MEL", "CRO"]),  # Storm vs Sharks
        ("2025-06-29 16:05", ["GLD", "NQL"]),  # Titans vs Cowboys
    ]
    
    locked_out_teams = set()
    for fixture_time, teams in fixtures:
        fixture_dt = datetime.strptime(fixture_time, '%Y-%m-%d %H:%M')
        if fixture_dt <= simulate_dt:
            locked_out_teams.update(teams)
    
    # Use the Team column from the main data instead of teamlists.csv
    locked_out_players = set()
    latest_round_data = consolidated_data.sort_values('Round').groupby('Player').last().reset_index()
    
    for team in locked_out_teams:
        team_players = latest_round_data[latest_round_data['Team'] == team]['Player'].tolist()
        locked_out_players.update(team_players)
    
    return locked_out_players


def is_player_locked(player_name: str, consolidated_data: pd.DataFrame, simulate_datetime: str) -> bool:
    """
    Check if a player is locked based on the simulated date/time.
    """
    if not simulate_datetime:
        return False
    
    # Parse the simulated date/time
    simulate_dt = datetime.strptime(simulate_datetime, '%Y-%m-%dT%H:%M')
    
    # Define the fixtures
    fixtures = [
        ("2025-06-26 19:50", ["PEN", "CBY"]),  # Panthers vs Bulldogs
        ("2025-06-27 18:00", ["MAN", "WST"]),  # Sea Eagles vs Tigers
        ("2025-06-27 20:00", ["NEW", "CAN"]),  # Knights vs Raiders
        ("2025-06-28 15:00", ["BRI", "WAR"]),  # Broncos vs Warriors
        ("2025-06-28 17:30", ["SGI", "PAR"]),  # Dragons vs Eels
        ("2025-06-28 19:50", ["DOL", "SOU"]),  # Dolphins vs Rabbitohs
        ("2025-06-29 14:00", ["MEL", "CRO"]),  # Storm vs Sharks
        ("2025-06-29 16:05", ["GLD", "NQL"]),  # Titans vs Cowboys
    ]
    
    locked_out_teams = set()
    for fixture_time, teams in fixtures:
        fixture_dt = datetime.strptime(fixture_time, '%Y-%m-%d %H:%M')
        if fixture_dt <= simulate_dt:
            locked_out_teams.update(teams)
    
    # Use the Team column from the main data instead of teamlists.csv
    latest_round_data = consolidated_data.sort_values('Round').groupby('Player').last().reset_index()
    player_data = latest_round_data[latest_round_data['Player'] == player_name]
    
    if player_data.empty:
        return False
        
    player_team = player_data['Team'].values[0]
    return player_team in locked_out_teams


def create_combination(players, total_price, salary_freed):
    """Helper function to create a trade combination dictionary"""
    return {
        'players': [create_player_dict(player) for player in players],
        'total_price': total_price,
        'total_projection': sum(player.get('Projection', 0) for player in players),
        'total_diff': sum(player.get('Diff', 0) for player in players),
        'salary_remaining': salary_freed - total_price
    }


def create_player_dict(player):
    """Helper function to create consistent player dictionary"""
    return {
        'name': player['Player'],
        'team': player['Team'],
        'position': player['POS1'],
        'secondary_position': player.get('POS2'),
        'price': player['Price'],
        'projection': player.get('Projection', 0),
        'diff': player.get('Diff', 0)
    }


def generate_trade_options(
    available_players: pd.DataFrame,
    salary_freed: float,
    maximize_base: bool = False,
    hybrid_approach: bool = False,
    max_options: int = 10,
    trade_type: str = 'likeForLike',
    traded_out_positions: List[str] = None,
    num_players_needed: int = 2
) -> List[Dict]:
    """
    Generate trade combinations based on selected optimization strategy while ensuring
    position requirements are met for both like-for-like and positional swap trades.
    """
    valid_combinations = []
    used_players = set()
    
    # Make a copy to avoid modifying the original DataFrame
    players_df = available_players.copy()
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Price', 'Diff', 'Projection']
    for col in numeric_columns:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
    
    # Convert DataFrame to list of dictionaries for easier manipulation
    players = players_df.to_dict('records')
    
    # Create a position mapping for each player
    position_mapping = {}
    for player in players:
        positions = [player['POS1']]
        if pd.notna(player.get('POS2')):
            positions.append(player['POS2'])
        position_mapping[player['Player']] = positions
    
    # Function to check if a player has at least one position from the required positions
    def has_valid_position(player, valid_positions):
        player_positions = position_mapping[player['Player']]
        return any(pos in valid_positions for pos in player_positions)
    
    # Function to check if a player combination works for like-for-like
    def is_valid_like_for_like_combo(player_combo):
        if not traded_out_positions or trade_type != 'likeForLike':
            return True
            
        # First check: each player must have at least one valid position
        if not all(has_valid_position(player, traded_out_positions) for player in player_combo):
            return False
            
        # Second check: all required positions must be covered
        positions_covered = set()
        for player in player_combo:
            for pos in position_mapping[player['Player']]:
                if pos in traded_out_positions:
                    positions_covered.add(pos)
        
        return set(traded_out_positions) == positions_covered
    
    # Function to handle position-balanced combinations for positional swap with exactly 2 positions
    def is_position_balanced_combo(first_player, second_player):
        if (trade_type == 'positionalSwap' and 
            traded_out_positions and len(traded_out_positions) == 2 and
            num_players_needed == 2):
            
            first_player_positions = position_mapping[first_player['Player']]
            second_player_positions = position_mapping[second_player['Player']]
            
            # Check if players cover both positions between them
            position_1_covered = any(pos == traded_out_positions[0] for pos in first_player_positions) or any(pos == traded_out_positions[0] for pos in second_player_positions)
            position_2_covered = any(pos == traded_out_positions[1] for pos in first_player_positions) or any(pos == traded_out_positions[1] for pos in second_player_positions)
            
            # Check if each player covers a different position
            first_player_covers_pos1 = any(pos == traded_out_positions[0] for pos in first_player_positions)
            first_player_covers_pos2 = any(pos == traded_out_positions[1] for pos in first_player_positions)
            
            second_player_covers_pos1 = any(pos == traded_out_positions[0] for pos in second_player_positions)
            second_player_covers_pos2 = any(pos == traded_out_positions[1] for pos in second_player_positions)
            
            # For a balanced combination, one player covering pos1 is needed and the other covering pos2
            return ((first_player_covers_pos1 and second_player_covers_pos2) or 
                    (first_player_covers_pos2 and second_player_covers_pos1))
        
        return True  # If not a position-balanced scenario, return True
    
    # Function to get price bracket for a player
    def get_price_bracket(price):
        brackets = [
            (250000, 317500),   # Bracket 1
            (317501, 385000),   # Bracket 2
            (385001, 452500),   # Bracket 3
            (452501, 520000),   # Bracket 4
            (520001, 587500),   # Bracket 5
            (587501, 655000),   # Bracket 6
            (655001, 722500),   # Bracket 7
            (722501, 790000),   # Bracket 8
            (790001, 857500)    # Bracket 9
        ]
        
        for i, (min_price, max_price) in enumerate(brackets):
            if min_price <= price <= max_price:
                return i + 1
        return None  # Price out of all defined brackets
    
    # Function to check if a player meets requirements for a specific level
    def meets_level_requirements(diff, price, level):
        bracket = get_price_bracket(price)
        if bracket is None:
            return False
        
        # Level 1 Requirements
        if level == 1:
            if bracket == 1 and diff >= 32.50: return True
            if bracket == 2 and 29.58 <= diff <= 32.49: return True
            if bracket == 3 and 26.67 <= diff <= 29.57: return True
            if bracket == 4 and 23.75 <= diff <= 26.66: return True
            if bracket == 5 and 20.83 <= diff <= 23.74: return True
            if bracket == 6 and 17.92 <= diff <= 20.82: return True
            if bracket == 7 and 15.00 <= diff <= 17.91: return True
            if bracket == 8 and 12.08 <= diff <= 14.99: return True
            if bracket == 9 and 9.17 <= diff <= 12.07: return True
            
        # Level 2 Requirements
        elif level == 2:
            if bracket == 1 and 29.58 <= diff <= 32.49: return True
            if bracket == 2 and 26.67 <= diff <= 29.57: return True
            if bracket == 3 and 23.75 <= diff <= 26.66: return True
            if bracket == 4 and 20.83 <= diff <= 23.74: return True
            if bracket == 5 and 17.92 <= diff <= 20.82: return True
            if bracket == 6 and 15.00 <= diff <= 17.91: return True
            if bracket == 7 and 12.08 <= diff <= 14.99: return True
            if bracket == 8 and 9.17 <= diff <= 12.07: return True
            if bracket == 9 and 7.80 <= diff <= 9.16: return True
            
        # Level 3 Requirements
        elif level == 3:
            if bracket == 1 and 26.67 <= diff <= 29.57: return True
            if bracket == 2 and 23.75 <= diff <= 26.66: return True
            if bracket == 3 and 20.83 <= diff <= 23.74: return True
            if bracket == 4 and 17.92 <= diff <= 20.82: return True
            if bracket == 5 and 15.00 <= diff <= 17.91: return True
            if bracket == 6 and 12.08 <= diff <= 14.99: return True
            if bracket == 7 and 9.17 <= diff <= 12.07: return True
            if bracket == 8 and 7.80 <= diff <= 9.16: return True
            # Bracket 9 excluded for level 3
            
        # Level 4 Requirements
        elif level == 4:
            if bracket == 1 and 23.75 <= diff <= 26.66: return True
            if bracket == 2 and 20.83 <= diff <= 23.74: return True
            if bracket == 3 and 17.92 <= diff <= 20.82: return True
            if bracket == 4 and 15.00 <= diff <= 17.91: return True
            if bracket == 5 and 12.08 <= diff <= 14.99: return True
            if bracket == 6 and 9.17 <= diff <= 12.07: return True
            if bracket == 7 and 7.80 <= diff <= 9.16: return True
            # Brackets 8 and 9 excluded for level 4
            
        # Level 5 Requirements
        elif level == 5:
            if bracket == 1 and 20.83 <= diff <= 23.74: return True
            if bracket == 2 and 17.92 <= diff <= 20.82: return True
            if bracket == 3 and 15.00 <= diff <= 17.91: return True
            if bracket == 4 and 12.08 <= diff <= 14.99: return True
            if bracket == 5 and 9.17 <= diff <= 12.07: return True
            if bracket == 6 and 7.80 <= diff <= 9.16: return True
            # Brackets 7, 8, and 9 excluded for level 5
            
        # Level 6 Requirements
        elif level == 6:
            if bracket == 1 and 17.92 <= diff <= 20.82: return True
            if bracket == 2 and 15.00 <= diff <= 17.91: return True
            if bracket == 3 and 12.08 <= diff <= 14.99: return True
            if bracket == 4 and 9.17 <= diff <= 12.07: return True
            if bracket == 5 and 7.80 <= diff <= 9.16: return True
            # Brackets 6, 7, 8, and 9 excluded for level 6
            
        # Level 7 Requirements
        elif level == 7:
            if bracket == 1 and 15.00 <= diff <= 17.91: return True
            if bracket == 2 and 12.08 <= diff <= 14.99: return True
            if bracket == 3 and 9.17 <= diff <= 12.07: return True
            if bracket == 4 and 7.80 <= diff <= 9.16: return True
            # Brackets 5, 6, 7, 8, and 9 excluded for level 7
            
        # Level 8 Requirements
        elif level == 8:
            if bracket == 1 and 12.08 <= diff <= 14.99: return True
            if bracket == 2 and 9.17 <= diff <= 12.07: return True
            if bracket == 3 and 7.80 <= diff <= 9.16: return True
            # Brackets 4, 5, 6, 7, 8, and 9 excluded for level 8
            
        # Level 9 Requirements
        elif level == 9:
            if bracket == 1 and 9.17 <= diff <= 12.07: return True
            if bracket == 2 and 7.80 <= diff <= 9.16: return True
            # Brackets 3, 4, 5, 6, 7, 8, and 9 excluded for level 9
            
        # Level 10 Requirements
        elif level == 10:
            if bracket == 1 and 7.80 <= diff <= 9.16: return True
            # All other brackets excluded for level 10
            
        return False
    
    # Function to calculate player priority
    def calculate_player_priority(player):
        diff = player['Diff']
        price = player['Price']
        
        # Players with upside below 7.80 points are excluded
        if diff < 7.80:
            return None
        
        # Check which level the player meets requirements for (starting from most valuable)
        for level in range(1, 11):
            if meets_level_requirements(diff, price, level):
                bracket = get_price_bracket(price)
                # Return tuple for sorting: (level, bracket, diff)
                # Lower level = higher priority, lower bracket = higher priority within level
                return (level, bracket, -diff)  # Negative diff so higher diff = higher priority
        
        # Player doesn't meet any level requirements
        return None
    
    # Sort players based on strategy
    if maximize_base:
        players.sort(key=lambda x: x['Projection'], reverse=True)
    elif hybrid_approach:
        # Calculate priority for each player and filter out players that don't meet any requirements
        players_with_priority = [(player, calculate_player_priority(player)) for player in players]
        valid_players = [p for p, priority in players_with_priority if priority is not None]
        valid_players.sort(key=lambda x: calculate_player_priority(x))
        players = valid_players
    else:  # maximize_value - use Diff
        players.sort(key=lambda x: x['Diff'], reverse=True)
    
    # Handle single player trades
    if num_players_needed == 1:
        for player in players:
            if player['Player'] in used_players:
                continue
            
            if player['Price'] <= salary_freed and (is_valid_like_for_like_combo([player]) or trade_type == 'positionalSwap'):
                combo = create_combination([player], player['Price'], salary_freed)
                valid_combinations.append(combo)
                used_players.add(player['Player'])
                if len(valid_combinations) >= max_options:
                    break
    # Handle 2+ player trades
    else:
        if maximize_base:
            # For 2+ player trades, find combinations with highest total Projection
            for i in range(len(players)):
                if players[i]['Player'] in used_players:
                    continue
                    
                first_player = players[i]
                
                # Find a valid second player
                for j in range(len(players)):
                    if j == i or players[j]['Player'] in used_players:
                        continue
                        
                    second_player = players[j]
                    
                    # Check if the combination is valid based on trade type
                    position_valid = (trade_type == 'likeForLike' and is_valid_like_for_like_combo([first_player, second_player])) or \
                                    (trade_type == 'positionalSwap' and is_position_balanced_combo(first_player, second_player))
                    
                    if not position_valid:
                        continue
                    
                    total_price = first_player['Price'] + second_player['Price']
                    if total_price <= salary_freed:
                        combo = create_combination([first_player, second_player], total_price, salary_freed)
                        valid_combinations.append(combo)
                        used_players.add(first_player['Player'])
                        used_players.add(second_player['Player'])
                        break  # Found a valid second player, move to next first player
                
                if len(valid_combinations) >= max_options:
                    break
                    
        elif hybrid_approach:
            # For hybrid approach, use the prioritized players in order
            for i, first_player in enumerate(players):
                if first_player['Player'] in used_players or first_player['Price'] > salary_freed:
                    continue
                
                remaining_salary = salary_freed - first_player['Price']
                
                # If using position-balanced approach, determine the needed complementary position
                if (trade_type == 'positionalSwap' and traded_out_positions and len(traded_out_positions) == 2):
                    # Identify which position the first player covers
                    first_player_positions = position_mapping[first_player['Player']]
                    first_player_pos1 = any(pos == traded_out_positions[0] for pos in first_player_positions)
                    
                    # Filter second player candidates to only those covering the other position
                    needed_position = traded_out_positions[1] if first_player_pos1 else traded_out_positions[0]
                    filtered_players = [p for p in players if 
                                      p['Player'] != first_player['Player'] and 
                                      p['Player'] not in used_players and
                                      (p['POS1'] == needed_position or 
                                       (pd.notna(p.get('POS2')) and p['POS2'] == needed_position))]
                    
                    second_player_candidates = filtered_players
                else:
                    # Use all available players if not using position-balanced approach
                    second_player_candidates = players
                
                # Find a valid second player
                found_match = False
                for second_player in second_player_candidates:
                    if second_player['Player'] == first_player['Player'] or second_player['Player'] in used_players:
                        continue
                    
                    # Check if the combination is valid based on trade type
                    position_valid = (trade_type == 'likeForLike' and is_valid_like_for_like_combo([first_player, second_player])) or \
                                    (trade_type == 'positionalSwap' and is_position_balanced_combo(first_player, second_player))
                    
                    if not position_valid:
                        continue
                    
                    total_price = first_player['Price'] + second_player['Price']
                    if total_price <= salary_freed:
                        combo = create_combination([first_player, second_player], total_price, salary_freed)
                        valid_combinations.append(combo)
                        used_players.add(first_player['Player'])
                        used_players.add(second_player['Player'])
                        found_match = True
                        break  # Found a valid second player, move to next first player
                
                if found_match and len(valid_combinations) >= max_options:
                    break
                    
        else:  # maximize_value - use Diff
            # For 2+ player trades, find combinations with highest total Diff
            for i, first_player in enumerate(players):
                if first_player['Player'] in used_players or first_player['Price'] > salary_freed:
                    continue
                
                # If using position-balanced approach, determine the needed complementary position
                if (trade_type == 'positionalSwap' and traded_out_positions and len(traded_out_positions) == 2):
                    # Identify which position the first player covers
                    first_player_positions = position_mapping[first_player['Player']]
                    first_player_pos1 = any(pos == traded_out_positions[0] for pos in first_player_positions)
                    
                    # Filter second player candidates to only those covering the other position
                    needed_position = traded_out_positions[1] if first_player_pos1 else traded_out_positions[0]
                    filtered_players = [p for p in players if 
                                       p['Player'] != first_player['Player'] and 
                                       p['Player'] not in used_players and
                                       (p['POS1'] == needed_position or 
                                        (pd.notna(p.get('POS2')) and p['POS2'] == needed_position))]
                    
                    second_player_candidates = filtered_players
                else:
                    # Use all available players if not using position-balanced approach
                    second_player_candidates = players
                
                # Find a valid second player
                found_match = False
                for j, second_player in enumerate(second_player_candidates):
                    if second_player['Player'] == first_player['Player'] or second_player['Player'] in used_players:
                        continue
                    
                    # Check if the combination is valid based on trade type
                    position_valid = (trade_type == 'likeForLike' and is_valid_like_for_like_combo([first_player, second_player])) or \
                                    (trade_type == 'positionalSwap' and is_position_balanced_combo(first_player, second_player))
                    
                    if not position_valid:
                        continue
                    
                    total_price = first_player['Price'] + second_player['Price']
                    if total_price <= salary_freed:
                        combo = create_combination([first_player, second_player], total_price, salary_freed)
                        valid_combinations.append(combo)
                        used_players.add(first_player['Player'])
                        used_players.add(second_player['Player'])
                        found_match = True
                        break  # Found a valid second player, move to next first player
                
                if found_match and len(valid_combinations) >= max_options:
                    break
    
    # Sort the final combinations before returning
    if maximize_base:
        valid_combinations.sort(key=lambda x: x['total_projection'], reverse=True)
    elif hybrid_approach:
        # For hybrid, combinations are already prioritized by player selection
        pass
    else:  # maximize_value
        valid_combinations.sort(key=lambda x: x['total_diff'], reverse=True)
    
    return valid_combinations[:max_options]


def calculate_trade_options(
    consolidated_data: pd.DataFrame,
    traded_out_players: List[str],
    maximize_base: bool = False,
    hybrid_approach: bool = False,
    max_options: int = 10,
    allowed_positions: List[str] = None,
    trade_type: str = 'likeForLike',
    min_games: int = 2,
    team_list: List[str] = None,
    simulate_datetime: str = None,
    apply_lockout: bool = False,
    excluded_players: List[str] = None,
    cash_in_bank: int = 0
) -> List[Dict]:
    """
    Calculate trade options based on the selected strategy.
    
    Parameters:
    consolidated_data (pd.DataFrame): DataFrame containing all player data
    traded_out_players (List[str]): List of player names being traded out
    maximize_base (bool): Whether to maximize base stats (Projection) instead of value (Diff)
    hybrid_approach (bool): Whether to use a hybrid approach combining value and base stats
    max_options (int): Maximum number of trade options to return
    allowed_positions (List[str]): List of positions to consider for positional swap
    trade_type (str): Type of trade ('likeForLike' or 'positionalSwap')
    min_games (int): Minimum number of games required for a player to be considered
    team_list (List[str]): Optional list of players to restrict trades to
    simulate_datetime (str): Optional datetime string for lockout simulation
    apply_lockout (bool): Whether to apply lockout restrictions
    excluded_players (List[str]): Optional list of players to exclude from trade options
    cash_in_bank (int): Additional cash to add to the salary freed up
    
    Returns:
    List[Dict]: List of trade option dictionaries
    """
    # Get locked out players if lockout restriction is applied
    locked_out_players = set()
    if apply_lockout:
        locked_out_players = get_locked_out_players(simulate_datetime, consolidated_data)
    
    # Get positions based on trade type
    if trade_type == 'likeForLike':
        traded_out_positions = get_traded_out_positions(traded_out_players, consolidated_data)
        positions_to_use = None
    elif trade_type == 'positionalSwap':
        traded_out_positions = allowed_positions  # Use selected positions for positional swap
        positions_to_use = allowed_positions
    else:
        traded_out_positions = None
        positions_to_use = None
    
    latest_round = consolidated_data['Round'].max()
    
    # Get number of players needed based on traded out players
    num_players_needed = len(traded_out_players)
    
    # Calculate total salary freed up from traded out players
    salary_freed = cash_in_bank  # Start with cash in bank value
    for player in traded_out_players:
        player_data = consolidated_data[consolidated_data['Player'] == player].sort_values('Round', ascending=False)
        if not player_data.empty:
            salary_freed += player_data.iloc[0]['Price']
        else:
            print(f"Warning: Could not find price data for {player}")
    
    print(f"Total salary freed up: ${salary_freed:,} (including ${cash_in_bank:,} cash in bank)")
    
    # Get all players from the latest round
    latest_round_data = consolidated_data[consolidated_data['Round'] == latest_round]
    available_players = latest_round_data[~latest_round_data['Player'].isin(traded_out_players)]
    
    # Apply excluded players filter
    if excluded_players and len(excluded_players) > 0:
        available_players = available_players[~available_players['Player'].isin(excluded_players)]
        if available_players.empty:
            print("Warning: No players available after excluding selected players")
            return []
    
    # Apply team list restriction if enabled
    if team_list:
        available_players = available_players[available_players['Player'].isin(team_list)]
        if available_players.empty:
            print("Warning: No players available after applying team list restriction")
            return []
    
    # Apply lockout restriction if enabled
    if apply_lockout:
        available_players = available_players[~available_players['Player'].isin(locked_out_players)]
        if available_players.empty:
            print("Warning: No players available after applying lockout restriction")
            return []
    
    # Filter players by allowed positions if specified
    if positions_to_use:
        if trade_type == 'positionalSwap':
            # Check both POS1 and POS2 for allowed positions
            mask = (
                available_players['POS1'].isin(positions_to_use) |
                available_players['POS2'].fillna('').isin(positions_to_use)
            )
            available_players = available_players[mask]
        else:
            # Like-for-like uses only POS1
            available_players = available_players[available_players['POS1'].isin(positions_to_use)]
        if available_players.empty:
            print("Warning: No players available with selected positions")
            return []
    
    # Generate trade options based on the selected strategy
    options = generate_trade_options(
        available_players,
        salary_freed,
        maximize_base,
        hybrid_approach,
        max_options,
        trade_type,
        traded_out_positions,
        num_players_needed
    )
    
    return options[:max_options]


if __name__ == "__main__":
    try:
        consolidated_data = load_data()
        print(f"Successfully loaded data for {consolidated_data['Round'].nunique()} rounds")
        
        # Get user preference for optimization strategy first
        while True:
            strategy = input("\nDo you want to:\n1. Maximize value (Diff)\n2. Maximize base stats (Projection)\n3. Hybrid approach (Diff + Projection)\nEnter 1, 2, or 3: ")
            if strategy in ['1', '2', '3']:
                break
            print("Invalid input. Please enter 1, 2, or 3.")

        maximize_base = (strategy == '2')
        hybrid_approach = (strategy == '3')

        # Then get position preferences
        valid_positions = ['HOK', 'HLF', 'CTR', 'WFB', 'EDG', 'MID']
        while True:
            print("\nSelect positions to consider:")
            print("0. All positions")
            for i, pos in enumerate(valid_positions, 1):
                print(f"{i}. {pos}")
            
            try:
                pos1 = int(input("\nSelect first position (0-6): "))
                if pos1 < 0 or pos1 > 6:
                    raise ValueError
                
                if pos1 == 0:
                    allowed_positions = None
                    break
                
                pos2 = int(input("Select second position (1-6, or same as first position): "))
                if pos2 < 1 or pos2 > 6:
                    raise ValueError
                
                allowed_positions = [valid_positions[pos1-1]]
                if pos1 != pos2:
                    allowed_positions.append(valid_positions[pos2-1])
                break
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
        
        # Get players to trade out
        player1 = input("\nEnter first player to trade out: ")
        player2 = input("Enter second player to trade out (leave blank for single player trade): ")
        
        traded_out_players = [player1]
        if player2:
            traded_out_players.append(player2)
        
        # Get lockout and simulation date/time preferences
        apply_lockout = input("Apply lockout restriction? (yes/no): ").strip().lower() == 'yes'
        simulate_datetime = None
        if apply_lockout:
            simulate_datetime = input("Enter simulated date/time (YYYY-MM-DDTHH:MM): ").strip()
        
        print(f"\nCalculating trade options for trading out: {', '.join(traded_out_players)}")
        print(f"Strategy: {'Maximizing base stats (Projection)' if maximize_base else 'Maximizing value (Diff)' if not hybrid_approach else 'Hybrid approach (Diff + Projection)'}")
        if allowed_positions:
            print(f"Considering only positions: {', '.join(allowed_positions)}")
        else:
            print("Considering all positions")
        
        options = calculate_trade_options(
            consolidated_data,
            traded_out_players,
            maximize_base=maximize_base,
            hybrid_approach=hybrid_approach,
            max_options=10,
            allowed_positions=allowed_positions,
            simulate_datetime=simulate_datetime,
            apply_lockout=apply_lockout
        )
        
        if options:
            print("\n=== Recommended Trade Combinations ===\n")
            for i, option in enumerate(options, 1):
                print(f"\nOption {i}")
                print("Players to trade in:")
                for player in option['players']:
                    if maximize_base:
                        print(f"- {player['name']} ({player['position']})")
                        print(f"  Team: {player['team']}")
                        print(f"  Price: ${player['price']:,}")
                        print(f"  Projected score: {player['projection']:.1f}")
                    else:
                        print(f"- {player['name']} ({player['position']})")
                        print(f"  Team: {player['team']}")
                        print(f"  Price: ${player['price']:,}")
                        print(f"  Upside: {player['diff']:.1f}")
                
                print(f"Total Price: ${option['total_price']:,}")
                if maximize_base:
                    print(f"Combined Projected score: {option['total_projection']:.1f}")
                else:
                    print(f"Combined Upside: {option['total_diff']:.1f}")
                print(f"Salary Remaining: ${option['salary_remaining']:,}")
            
    except FileNotFoundError:
        print("Error: Could not find data file in the current directory")
    except ValueError as e:
        print("Error:", str(e))
    except Exception as e:
        print("An error occurred:", str(e))