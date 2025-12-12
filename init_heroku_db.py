import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from db_operations import get_column_definitions  # Reuse existing column definitions

# Load environment variables from project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

def init_heroku_database():
    # Get the database URL from Heroku environment
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not found")
        
    # Handle Heroku's postgres:// URL format
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    # Create SQLAlchemy engine
    engine = create_engine(database_url)
    
    try:
        # Read the Excel file
        df = pd.read_excel('NRL_stats.xlsx')
        
        # Define the required columns using exact names from Excel
        required_columns = [
            'Round',
            'Player',
            'Team',
            'POS1',
            'POS2',
            'Price',
            'Priced at',
            'Projection',
            'Diff',
            'Bye Round Grading',
            'Injured'
        ]
        
        # Filter DataFrame to include only required columns
        df = df[required_columns].copy()
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Convert numeric columns and handle NaN values
        numeric_columns = ['Round', 'Price', 'Priced_at', 'Projection', 'Diff', 'Bye_Round_Grading']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].where(pd.notnull(df[col]), None)

        def normalize_injured(value):
            """Convert the Injured column to booleans, preserving blanks as None."""
            if pd.isna(value):
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(int(value))
            val = str(value).strip().lower()
            if val in {'true', 'yes', 'y', '1', '1.0'}:
                return True
            if val in {'false', 'no', 'n', '0', '0.0'}:
                return False
            return None

        if 'Injured' in df.columns:
            df['Injured'] = df['Injured'].apply(normalize_injured)
        
        # Replace any remaining NaN values with None
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        
        # Create table with matching schema
        with engine.connect() as connection:
            # Drop table if exists
            connection.execute(text("DROP TABLE IF EXISTS player_stats"))
            
            # Get column definitions using existing function
            column_defs = get_column_definitions(df)
            create_table_sql = f"""
            CREATE TABLE player_stats (
                id SERIAL PRIMARY KEY,
                {','.join(column_defs)}
            )
            """
            connection.execute(text(create_table_sql))
            connection.commit()
            
            # Insert data using SQLAlchemy's to_sql
            df.to_sql('player_stats', engine, if_exists='append', index=False)
            
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    init_heroku_database()