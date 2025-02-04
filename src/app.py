from flask import Flask, render_template, request, jsonify
from nrl_trade_calculator import calculate_trade_options, load_data, assign_priority_level, is_player_locked
from typing import List, Dict, Any
import traceback
import pandas as pd

app = Flask(__name__)

def prepare_trade_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare trade option data for JSON response, ensuring all required fields exist
    and are properly formatted.
    """
    prepared_option = {
        'players': [],
        'total_price': int(option.get('total_price', 0)),
        'salary_remaining': int(option.get('salary_remaining', 0)),
        'total_base': float(option.get('total_base', 0)),
        'total_base_premium': float(option.get('total_base_premium', 0)),
        'total_avg_base': float(option.get('total_avg_base', 0)) if 'total_avg_base' in option else 0.0,
        'combo_avg_bpre': float(option.get('combo_avg_bpre', 0)) if 'combo_avg_bpre' in option else 0.0
    }

    for player in option.get('players', []):
        prepared_player = {
            'name': player.get('name', ''),
            'position': player.get('position', ''),
            'team': player.get('team', ''),  
            'price': int(player.get('price', 0)),
            'total_base': float(player.get('total_base', 0)),
            'base_premium': int(float(player.get('base_premium', 0))),
            'consecutive_good_weeks': int(player.get('consecutive_good_weeks', 0)),
            'avg_base': float(player.get('avg_base', 0)) if 'avg_base' in player else 0.0
        }
        
        if 'priority_level' in player:
            prepared_player['priority_level'] = int(player['priority_level'])
            
        prepared_option['players'].append(prepared_player)

    return prepared_option

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_player_lockout', methods=['POST'])
def check_player_lockout():
    try:
        player_name = request.form['player_name']
        simulate_datetime = request.form.get('simulateDateTime')
        
        file_path = "NRL_stats.xlsx"
        consolidated_data = load_data(file_path)
        
        is_locked = is_player_locked(player_name, consolidated_data, simulate_datetime)
        
        return jsonify({
            'is_locked': is_locked
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def simulate_rule_levels(consolidated_data: pd.DataFrame, rounds: List[int]) -> None:
    player_name = consolidated_data['Player'].unique()[0]  # Assuming the first player in the data

    rule_descriptions = {
        1: "BPRE >= 14 for last 3 weeks",
        2: "BPRE >= 21 for last 2 weeks",
        3: "2 week Average BPRE >= 26",
        4: "BPRE >= 12 for last 3 weeks",
        5: "BPRE >= 19 for last 2 weeks",
        6: "2 week Average BPRE >= 24",
        7: "BPRE >= 10 for last 3 weeks",
        8: "BPRE >= 17 for last 2 weeks",
        9: "2 week Average BPRE >= 22",
        10: "BPRE >= 8 for last 3 weeks",
        11: "BPRE >= 15 for last 2 weeks",
        12: "2 week Average BPRE >= 20",
        13: "BPRE >= 6 for last 3 weeks",
        14: "BPRE >= 13 for last 2 weeks",
        15: "2 week Average BPRE >= 18",
        16: "BPRE >= 10 for last 2 weeks",
        17: "2 week Average BPRE >= 15",
        18: "BPRE >= 8 for last 2 weeks",
        19: "2 week Average BPRE >= 13",
        20: "BPRE >= 6 for last 2 weeks",
        21: "2 week Average BPRE >= 11",
        22: "BPRE >= 2 for last 3 weeks",
        23: "BPRE >= 4 for last 2 weeks",
        24: "2 week Average BPRE >= 9",
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
        player1 = request.form['player1']
        player2 = request.form.get('player2')
        strategy = request.form['strategy']
        trade_type = request.form['tradeType']
        allowed_positions = request.form.getlist('positions') if trade_type == 'positionalSwap' else []
        restrict_to_team_list = 'restrictToTeamList' in request.form
        apply_lockout = 'applyLockout' in request.form
        simulate_datetime = request.form.get('simulateDateTime')

        file_path = "NRL_stats.xlsx"
        consolidated_data = load_data(file_path)

        team_list = None
        if restrict_to_team_list:
            team_list_path = "teamlists.csv"
            team_list = load_data(team_list_path)['Player'].unique().tolist()

        maximize_base = (strategy == '2')
        hybrid_approach = (strategy == '3')

        traded_out_players = [player1]
        if player2:
            traded_out_players.append(player2)

        # Validate lockout status for traded out players
        if apply_lockout:
            locked_players = []
            for player in traded_out_players:
                if is_player_locked(player, consolidated_data, simulate_datetime):
                    locked_players.append(player)
            
            if locked_players:
                return jsonify({
                    'error': f"{' and '.join(locked_players)}'s lockout has expired"
                }), 400

        options = calculate_trade_options(
            consolidated_data=consolidated_data,
            traded_out_players=traded_out_players,
            maximize_base=maximize_base,
            hybrid_approach=hybrid_approach,
            max_options=10,
            allowed_positions=allowed_positions,
            trade_type=trade_type,
            team_list=team_list,
            simulate_datetime=simulate_datetime,
            apply_lockout=apply_lockout
        )

        prepared_options = [prepare_trade_option(option) for option in options]

        return jsonify(prepared_options)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error occurred: {error_traceback}")
        
        return jsonify({
            'error': f'An error occurred while calculating trade options: {str(e)}'
        }), 500

@app.route('/players', methods=['GET'])
def get_players():
    try:
        file_path = "NRL_stats.xlsx"
        consolidated_data = load_data(file_path)
        player_names = consolidated_data['Player'].unique().tolist()
        return jsonify(player_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    try:
        while True:
            choice = input("\nDo you want to:\n1. Run the ordinary trade calculator\n2. Run rule set simulation for 1 player\nEnter 1 or 2: ")
            if choice in ['1', '2']:
                break
            print("Invalid input. Please enter 1 or 2.")

        file_path = "NRL_stats.xlsx" if choice == '1' else "player_simulation.xlsx"
        consolidated_data = load_data(file_path)
        print(f"Successfully loaded data for {consolidated_data['Round'].nunique()} rounds")

        if choice == '2':
            rounds = list(range(1, int(consolidated_data['Round'].max()) + 1))
            simulate_rule_levels(consolidated_data, rounds)
        else:
            app.run(debug=True)

    except FileNotFoundError:
        print("Error: Could not find data file in the current directory")
    except ValueError as e:
        print("Error:", str(e))
    except Exception as e:
        print("An error occurred:", str(e))

# comment