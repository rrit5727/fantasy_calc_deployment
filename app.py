from flask import Flask, render_template, request, jsonify
from nrl_trade_calculator import calculate_trade_options, load_data, assign_priority_level, is_player_locked
from typing import List, Dict, Any
import traceback
import pandas as pd
import os
from datetime import datetime

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
        
        file_path = os.path.join(os.path.dirname(__file__), "NRL_stats.xlsx")
        consolidated_data = load_data(file_path)
        
        is_locked = is_player_locked(player_name, consolidated_data, simulate_datetime)
        
        return jsonify({
            'is_locked': is_locked
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

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

        file_path = os.path.join(os.path.dirname(__file__), "NRL_stats.xlsx")
        consolidated_data = load_data(file_path)

        team_list = None
        if restrict_to_team_list:
            team_list_path = os.path.join(os.path.dirname(__file__), "teamlists.csv")
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
        file_path = os.path.join(os.path.dirname(__file__), "NRL_stats.xlsx")
        consolidated_data = load_data(file_path)
        player_names = consolidated_data['Player'].unique().tolist()
        return jsonify(player_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)