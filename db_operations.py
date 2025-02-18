import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Database connection parameters from the .env file
DB_PARAMS = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT')
}

def create_db_connection():
    """Create database connection"""
    return psycopg2.connect(**DB_PARAMS)

def get_column_definitions(df):
    """Generate SQL column definitions based on DataFrame columns"""
    column_types = {
        'Round': 'INTEGER',
        'Player': 'VARCHAR(100)',
        'Team': 'VARCHAR(50)',
        'Age': 'INTEGER',
        'POS1': 'VARCHAR(10)',
        'POS2': 'VARCHAR(10)',
        'Price': 'DECIMAL(12,2)',
        'Priced_at': 'DECIMAL(12,4)',
        'PTS': 'DECIMAL(12,2)',
        'Total_base': 'DECIMAL(12,2)',
        'Base_exceeds_price_premium': 'DECIMAL(12,4)'
    }
    
    # Default type for any column not explicitly defined
    default_type = 'VARCHAR(100)'
    columns = []
    for col in df.columns:
        clean_col = col.strip().replace(' ', '_')
        col_type = column_types.get(clean_col, default_type)
        columns.append(f'"{clean_col}" {col_type}')
    return columns

def create_table(conn, df):
    """Create the table with columns matching the DataFrame"""
    column_defs = get_column_definitions(df)
    create_table_sql = f"""
    DROP TABLE IF EXISTS player_stats;
    CREATE TABLE player_stats (
        id SERIAL PRIMARY KEY,
        {','.join(column_defs)}
    );
    """
    
    with conn.cursor() as cur:
        cur.execute(create_table_sql)
    conn.commit()

def import_excel_data(excel_file_path):
    """Import data from Excel to PostgreSQL"""
    # Read Excel file
    df = pd.read_excel(excel_file_path)
    
    # Print available columns for debugging
    print("Available columns in Excel file:")
    for col in df.columns:
        print(f"- '{col}'")
    
    # Define the required columns using exact names from Excel
    required_columns = [
        'Round', 'Player', 'Team', 'Age', 'POS1', 'POS2',
        'Price', 'Priced at', 'PTS', 'Total base',
        'Base exceeds price premium'
    ]
    
    # Filter DataFrame to include only required columns
    try:
        df = df[required_columns].copy()
    except KeyError as e:
        missing_cols = [col for col in required_columns if col not in df.columns]
        print("\nMissing columns:")
        for col in missing_cols:
            print(f"- '{col}'")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean column names after filtering
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Convert numeric columns to appropriate types and handle NaN values
    numeric_columns = ['Round', 'Age', 'Price', 'Priced_at', 'PTS', 
                      'Total_base', 'Base_exceeds_price_premium']
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to float first to handle any decimal values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace NaN with None
            df[col] = df[col].where(pd.notnull(df[col]), None)
    
    # Replace any remaining NaN values with None
    df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
    
    # Create table with matching columns
    conn = create_db_connection()
    create_table(conn, df)
    
    # Insert data using cursor
    with conn.cursor() as cur:
        for idx, row in df.iterrows():
            # Convert row to list and handle any remaining NaN values
            row_values = [None if pd.isna(val) else val for val in row]
            columns = ','.join(f'"{col}"' for col in df.columns)
            values = ','.join('%s' for _ in df.columns)
            insert_sql = f'INSERT INTO player_stats ({columns}) VALUES ({values})'
            try:
                cur.execute(insert_sql, row_values)
            except Exception as e:
                print(f"Error at row {idx+1}")
                print(f"Values being inserted: {row_values}")
                print(f"Error details: {str(e)}")
                conn.rollback()  # Rollback the transaction
                raise e
    
    conn.commit()
    conn.close()

def main():
    try:
        # Import data
        import_excel_data('NRL_stats.xlsx')
        print("Data import completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()