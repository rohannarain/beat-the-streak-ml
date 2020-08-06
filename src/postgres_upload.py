from sqlalchemy import create_engine
import pandas as pd 
import os
from dotenv import load_dotenv

load_dotenv()

def connect_to_postgres():
	engine = create_engine(os.getenv("DATABASE_URL"), 
						connect_args={'sslmode': 'require'})
	return engine.connect()

def to_predictions_db(df):
	with connect_to_postgres() as connection:
		df.to_sql('predictions', con=connection, if_exists='append', index=False)

def create_inspector(engine):
	return inspect(engine)

def check_total_rows(engine, inspector):
	row_count = 0
	for table in inspector.get_table_names():
		row_count += engine.execute(f"SELECT COUNT(*) FROM {table}").scalar()
	return row_count




