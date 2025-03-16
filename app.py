from flask import Flask, render_template, request, jsonify
from flask_caching import Cache  # Added for caching
from nrl_trade_calculator import calculate_trade_options, load_data, is_player_locked
from typing import List, Dict, Any
import traceback
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)
CACHE_TIMEOUT = 300  # 5 minutes cache

# Add these global variables at the top of the file, after the app initialization
_cached_data = None
_last_cache_time = 0

def prepare_trade_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare trade option for JSON response with the new data structure.
    """
    players = []
    total_price = int(option.get('total_price', 0))
    
    # Precompute player data
    for player in option.get('players', []):
        p = {
            'name': player.get('name', ''),
            'position': player.get('position', ''),
            'team': player.get('team', ''),
            'price': int(player['price'])
        }
        
        # Add diff or projection based on what's available
        if 'diff' in player:
            p['diff'] = float(player['diff'])
        if 'projection' in player:
            p['projection'] = float(player['projection'])
            
        players.append(p)

    result = {
        'players': players,
        'total_price': total_price,
        'salary_remaining': int(option['salary_remaining'])
    }
    
    # Add total metrics based on what's available
    if 'total_diff' in option:
        result['total_diff'] = float(option['total_diff'])
    if 'total_projection' in option:
        result['total_projection'] = float(option['total_projection'])
        
    return result

@app.route('/')
def index():
    hotjar_id = os.getenv('HOTJAR_ID')
    return render_template('index.html', hotjar_id=hotjar_id)

@app.route('/check_player_lockout', methods=['POST'])
def check_player_lockout():
    try:
        player_name = request.form['player_name']
        simulate_datetime = request.form.get('simulateDateTime')
        
        # Use cached data
        consolidated_data = cached_load_data()
        
        is_locked = is_player_locked(player_name, consolidated_data, simulate_datetime)
        
        return jsonify({'is_locked': is_locked})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Added cached data loading
@cache.cached(timeout=CACHE_TIMEOUT, key_prefix='load_data')
def cached_load_data():
    """
    Load data from database with caching to improve performance.
    """
    global _cached_data
    global _last_cache_time
    
    # Check if we need to refresh the cache (every 15 minutes)
    current_time = time.time()
    if _cached_data is None or current_time - _last_cache_time > 900:  # 15 minutes in seconds
        try:
            _cached_data = load_data()
            _last_cache_time = current_time
            app.logger.info(f"Data cache refreshed with {len(_cached_data)} records")
        except Exception as e:
            app.logger.error(f"Error refreshing data cache: {str(e)}")
            # If we have cached data, use it even if it's stale
            if _cached_data is not None:
                app.logger.warning("Using stale cached data due to refresh error")
            else:
                # No cached data available, must raise the error
                raise
    
    return _cached_data

def simulate_rule_levels(consolidated_data: pd.DataFrame, rounds: List[int]) -> None:
    # Existing implementation unchanged
    player_name = consolidated_data['Player'].unique()[0]

    rule_descriptions = {
        1: "BPRE >= 14 for last 3 weeks",
        # ... rest of rule descriptions unchanged ...
        25: "No rules satisfied"
    }

    for round_num in rounds:
        recent_rounds = sorted(consolidated_data['Round'].unique())
        recent_rounds = [r for r in recent_rounds if r <= round_num][-4:]
        cumulative_data = consolidated_data[consolidated_data['Round'].isin(recent_rounds)]
        player_data = cumulative_data[cumulative_data['Player'] == player_name]
        
        if player_data.empty:
            print(f"Round {round_num}: No data for player {player_name}")
            continue
        
        priority_level = assign_priority_level(player_data.iloc[-1], cumulative_data)
        rule_description = rule_descriptions.get(priority_level, "Unknown rule")
        print(f"Rule levels passed as at round {round_num}: Rule Level Satisfied: {priority_level} - {rule_description}")

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # Extract form data first before any processing
        form_data = {
            'player1': request.form['player1'],
            'player2': request.form.get('player2'),
            'strategy': request.form['strategy'],
            'tradeType': request.form['tradeType'],
            'positions': request.form.getlist('positions') if request.form['tradeType'] == 'positionalSwap' else [],
            'restrictToTeamList': 'restrictToTeamList' in request.form,
            'applyLockout': 'applyLockout' in request.form,
            'simulateDateTime': request.form.get('simulateDateTime'),
            'excludedPlayers': request.form.getlist('excludedPlayers'),  # Get all excluded players
            'cashInBank': int(request.form.get('cashInBank', 0))  # Get cash in bank with default of 0
        }

        # Use cached data load
        consolidated_data = cached_load_data()

        # Early validation for required players
        traded_out_players = [form_data['player1']]
        if form_data['player2']:
            traded_out_players.append(form_data['player2'])

        # Validate lockout status first to avoid unnecessary processing
        if form_data['applyLockout']:
            locked_players = []
            for player in traded_out_players:
                if is_player_locked(player, consolidated_data, form_data['simulateDateTime']):
                    locked_players.append(player)
            
            if locked_players:
                return jsonify({
                    'error': f"{' and '.join(locked_players)}'s lockout has expired"
                }), 400

        # Load team list if needed
        team_list = None
        if form_data['restrictToTeamList']:
            team_list_path = "teamlists.csv"
            team_df = pd.read_csv(team_list_path)
            team_list = team_df['Player'].unique().tolist()

        # Strategy flags
        strategy = form_data['strategy']
        maximize_base = (strategy == '2')
        hybrid_approach = (strategy == '3')

        # Calculate trade options
        options = calculate_trade_options(
            consolidated_data,
            traded_out_players,
            maximize_base=maximize_base,
            hybrid_approach=hybrid_approach,
            max_options=10,
            allowed_positions=form_data['positions'] if form_data['positions'] else None,
            trade_type=form_data['tradeType'],
            team_list=team_list,
            simulate_datetime=form_data['simulateDateTime'],
            apply_lockout=form_data['applyLockout'],
            excluded_players=form_data['excludedPlayers'],
            cash_in_bank=form_data['cashInBank']  # Pass the cash in bank value
        )

        # Format options for frontend
        formatted_options = []
        for option in options:
            formatted_option = {
                'players': [],
                'totalPrice': option['total_price'],
                'salaryRemaining': option['salary_remaining']
            }
            
            if maximize_base:
                formatted_option['totalProjection'] = option['total_projection']
            else:
                formatted_option['totalDiff'] = option['total_diff']
                
            for player in option['players']:
                # Check if the player has a secondary position that isn't None or empty
                position_display = player['position']
                if player.get('secondary_position') and pd.notna(player.get('secondary_position')):
                    position_display = f"{player['position']}/{player['secondary_position']}"
                
                player_info = {
                    'name': player['name'],
                    'team': player['team'],
                    'position': position_display,
                    'price': player['price']
                }
                
                if maximize_base:
                    player_info['projection'] = player['projection']
                else:
                    player_info['diff'] = player['diff']
                    
                formatted_option['players'].append(player_info)
            
            formatted_options.append(formatted_option)
        
        # Return the array directly to match what the frontend expects
        return jsonify(formatted_options)
        
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error in calculate: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

@app.route('/players', methods=['GET'])
def get_players():
    try:
        # Use cached data
        consolidated_data = cached_load_data()
        player_names = consolidated_data['Player'].unique().tolist()
        return jsonify(player_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_player_names_with_prices', methods=['GET'])
def get_player_names_with_prices():
    try:
        # Load data
        consolidated_data = cached_load_data()
        
        # Get latest round data
        latest_round = consolidated_data['Round'].max()
        latest_data = consolidated_data[consolidated_data['Round'] == latest_round]
        
        # Create list of players with prices
        player_data = []
        for _, row in latest_data.iterrows():
            player_data.append({
                'label': row['Player'],
                'value': row['Player'],
                'price': str(int(row['Price'] / 1000))  # Convert to k format
            })
        
        return jsonify(player_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def init_app(app):
    try:
        with app.app_context():
            # Check if we're in development environment
            is_development = os.getenv('FLASK_ENV') == 'development'
            
            if is_development:
                db_params = {
                    'host': os.getenv('DB_HOST'),
                    'database': os.getenv('DB_DATABASE'),
                    'user': os.getenv('DB_USER'),
                    'password': os.getenv('DB_PASSWORD'),
                    'port': os.getenv('DB_PORT')
                }
                conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
            else:
                database_url = os.getenv("DATABASE_URL")
                if database_url:
                    if database_url.startswith("postgres://"):
                        database_url = database_url.replace("postgres://", "postgresql://", 1)
                    conn_str = database_url
                else:
                    raise ValueError("DATABASE_URL not found in production environment")
            
            engine = create_engine(conn_str)
            
            with engine.connect() as connection:
                # Check if table exists
                result = connection.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'player_stats'
                    );
                """))
                table_exists = result.scalar()
                
                if not table_exists:
                    # Initialize database
                    init_heroku_database()
    except Exception as e:
        print(f"Error checking/initializing database: {str(e)}")
        raise

# Initialize the database when the app starts
init_app(app)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
    try:
        while True:
            choice = input("\nDo you want to:\n1. Run the ordinary trade calculator\n2. Run rule set simulation for 1 player\nEnter 1 or 2: ")
            if choice in ['1', '2']:
                break
            print("Invalid input. Please enter 1 or 2.")

        # Use cached data load
        consolidated_data = cached_load_data()
        
        if consolidated_data.empty:
            raise ValueError("No data loaded from database")
            
        print(f"Successfully loaded data for {consolidated_data['Round'].nunique()} rounds")

        if choice == '2':
            player_name = input("Enter player name for simulation: ")
            if player_name not in consolidated_data['Player'].unique():
                raise ValueError(f"Player {player_name} not found in database")
            
            player_data = consolidated_data[consolidated_data['Player'] == player_name]
            rounds = list(range(1, int(consolidated_data['Round'].max()) + 1))
            simulate_rule_levels(player_data, rounds)
        else:
            app.run(debug=True)

    except ValueError as e:
        print("Error:", str(e))
    except Exception as e:
        print("An error occurred:", str(e))
        raise