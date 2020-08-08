from flask import render_template, redirect, flash, url_for, request
from app import app
from cloud_storage import download_blob, check_gcloud_blob_exists
from bs4 import BeautifulSoup

import csv
import requests

TITLE = 'Beat the Streak with ML'
TODAY = app.config['TODAY_PACIFIC_TIME']

def find_player_bbref(player_name):
	bbref = 'https://baseball-reference.com'
	last_initial = player_name.lower().split(" ")[-1][0]

	page_req = requests.get(f'{bbref}/players/{last_initial}/')
	soup = BeautifulSoup(page_req.text, 'html.parser')

	active_player_tags = soup.div.find_all('b')
	active_players = {tag.a.text:f'{bbref}{tag.a["href"]}' for tag in active_player_tags}

	return active_players.get(player_name,
		'Player not found. Either they are not an active player or they do not exist.')

def get_predictions_file():
	predictions_csv = f'data/predictions/season_{TODAY[-4:]}/predictions_{TODAY}.csv'
	print("Checking if there are predictions available for today...")
	blob_exists = check_gcloud_blob_exists(predictions_csv)
	print("Bucket checked")

	blob_not_found_msg = ""

	if blob_exists:
		download_blob(predictions_csv, 
			f"app/static/predictions_{TODAY}.csv")
	else: 
		blob_not_found_msg = "There are no predictions for today."

	return blob_not_found_msg

blob_not_found_msg = get_predictions_file()

def generate_predictions():
	display_cols = ['Name', 'Team', 'Hit Probability']
	col_names, cols_list = None, None

	with open(f"app/static/predictions_{TODAY}.csv", newline="") as csvfile:
		preds = csv.reader(csvfile)
		header = next(preds)
		col_names = {colname: idx for idx, colname in enumerate(header) if colname in display_cols}
		cols_list = list(zip(*[row for row in preds]))

	urls = []
	for player in cols_list[col_names['Name']]:
		urls.append(find_player_bbref(player))

	return col_names, cols_list, urls

col_names, cols_list, urls = (None, None, None) if blob_not_found_msg else generate_predictions()

@app.route("/")
@app.route("/home")
def home():
	return render_template('index.html', 
							blob_not_found_msg=blob_not_found_msg,
							col_names=col_names,
							cols_list=cols_list,
							urls=urls,
							today=TODAY.replace("_", "/"))

@app.route("/about")
def about():
	return render_template('about.html')

@app.route("/past_results")
def past_results():
	return render_template('past-results.html')
