import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
from db_operations import create_db_connection

# Load environment variables from .env.local
load_dotenv(dotenv_path=Path('.env.local'))

def create_team_lists_table(conn, df):
    """
    Create the team_lists table in PostgreSQL.
    Drops the table if it already exists.
    """
    create_table_sql = """
    DROP TABLE IF EXISTS team_lists;
    CREATE TABLE team_lists (
        id SERIAL PRIMARY KEY,
        "Player_Number" INTEGER,
        "Player" VARCHAR(100)
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()

def import_teamlists_data(excel_file_path):
    """
    Import data from the team lists Excel file into PostgreSQL.
    
    Expects the Excel file to have exactly two columns:
      - "Player Number"
      - "Player"
    """
    # Read Excel file
    df = pd.read_excel(excel_file_path)
    
    # Print available columns for debugging
    print("Available columns in Teamlists Excel file:")
    for col in df.columns:
        print(f"- '{col}'")
    
    # Define the required columns (using exact names as in the Excel file)
    required_columns = ['Player Number', 'Player']
    
    # Filter DataFrame to include only required columns
    try:
        df = df[required_columns].copy()
    except KeyError as e:
        missing_cols = [col for col in required_columns if col not in df.columns]
        print("\nMissing required columns:")
        for col in missing_cols:
            print(f"- '{col}'")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean column names: "Player Number" -> "Player_Number"
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Convert the Player_Number column to numeric
    if 'Player_Number' in df.columns:
        df['Player_Number'] = pd.to_numeric(df['Player_Number'], errors='coerce')
    
    # Create table and insert data
    conn = create_db_connection()
    create_team_lists_table(conn, df)
    
    with conn.cursor() as cur:
        for idx, row in df.iterrows():
            row_values = [None if pd.isna(val) else val for val in row]
            columns = ','.join(f'"{col}"' for col in df.columns)
            values_placeholders = ','.join('%s' for _ in df.columns)
            insert_sql = f'INSERT INTO team_lists ({columns}) VALUES ({values_placeholders})'
            cur.execute(insert_sql, row_values)
    conn.commit()
    conn.close()

def main():
    try:
        # Determine the file path for the team lists Excel file (assumed to be in the same directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        excel_file_path = os.path.join(current_dir, "teamlists.xlsx")
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"File not found: {excel_file_path}")
        
        import_teamlists_data(excel_file_path)
        print("Teamlists data import completed successfully!")
    except Exception as e:
        print(f"An error occurred while importing team lists data: {e}")

if __name__ == "__main__":
    main()
