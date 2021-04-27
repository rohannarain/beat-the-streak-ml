import pandas as pd
import statsapi
import json
import datetime
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
from collections import OrderedDict

import sys
import argparse
import os
from pathlib import Path
import time

from cloud_storage import *

from utils_config import Config 
yesterday = Config.get("YESTERDAY")
today = Config.get("TODAY")

CURR_SEASON = Config.get("curr_season")
CURR_SEASON_START = Config.get("curr_season_start")
CURR_SEASON_END = Config.get("curr_season_end")

################################################## UTILITY/GETTER FUNCTIONS #######################################################

def get_hitters_list(team_id: int, date=CURR_SEASON_END) -> list: 
    """
    A function that gets a list of all the hitters on a given team.
    
    Parameters 
    -----–-----------
    team_id: int
        The team ID number (i.e. 137 for S.F. Giants)
    """
    player_names = []
    roster = statsapi.roster(team_id, date=date)
    roster_list = roster.split("\n")[:-1]
    for player in roster_list:
        player_attrs = player.split()
        player_pos = player_attrs[1]
        if player_pos != "P":
            player_names.append(" ".join(player.split()[2:]))
    return player_names

def get_player_id_from_name(player_name: str, season=CURR_SEASON) -> int:
    """
    A function that gets the player ID for a name entered in any 
    format (Last, First; First Last; Last, etc.).
    
    Parameters 
    -----–-----------
    player_name: str
        The name of a player as a string (i.e. "Buster Posey")
    """
    try:
        return statsapi.lookup_player(player_name, season=season)[0]['id']
    except IndexError:
        return False

def check_pos_player(player_name: str, season=CURR_SEASON) -> bool:
    """
    A function that returns a bool indicating whether or not the player
    is a position player (as opposed to a pitcher).
    
    Parameters 
    -----–-----------
    player_name: str
        The name of a player as a string (i.e. "Buster Posey")
    """
    try:
        return statsapi.lookup_player(player_name, season=season)[0]['primaryPosition']['abbreviation'] != "P"
    except IndexError:
        return False

def get_current_season_stats(player_name: str, current_team: str, date: str, season=CURR_SEASON) -> bool:
    """
    One of the main data retrieval functions. Returns a dictionary 
    mapping the names of different statistics to the values of those
    statistics. Only includes overall season statistics for the player
    passed in. 
    
    Parameters 
    -----–-----------
    player_name: str
        The name of a player as a string (i.e. "Buster Posey")
    """

    if not check_pos_player(player_name):
        raise ValueError("Player name entered is not a position player")
    
    player_id = get_player_id_from_name(player_name)
    stats_dict = OrderedDict({"Name": player_name, "ID": player_id, "Team": current_team})
    
    # Look up the player's current season hitting stats
    stats_hydration = f'stats(group=[hitting],type=[byDateRange],startDate={CURR_SEASON_START},endDate={date},sportId=1)'
    get_player_stats = statsapi.get('person', {'personId': player_id, 'hydrate': stats_hydration})

    stats_dict.update(get_player_stats['people'][0]['stats'][0]['splits'][0]['stat'])
    
    return stats_dict


# These functions were defined with the help of toddrob99 on github, who developed the
# MLB-StatsAPI module. I made a post on reddit.com/r/mlbdata, which he mantains to 
# answer questions about making API calls for specific purposes. I asked how to get stats
# over the past x days and how to get head-to-head batting stats. The post is linked
# here: https://www.reddit.com/r/mlbdata/comments/cewwfo/getting_headtohead_batting_stats_and_last_x_games/?

def get_h2h_vs_pitcher(batter_id: int, opponent_id: int, season=CURR_SEASON) -> OrderedDict:
    """
    Returns a dictionary containing a limited amount of head-to-head batting 
    statistics between the hitter (batter_id) and pitcher (opponent_id) 
    specified. One of the main data retrieval functions.
    
    Parameters 
    -----–-----------
    batter_id: int
        The 6-digit ID of a batter, which can be fetched using 
        get_player_id_from_name('Hitter Name').
    
    opponent_id: int
        The 6-digit ID of a pitcher, which can be fetched using 
        get_player_id_from_name('Pitcher Name').
    """
    
    hydrate = f'stats(group=[hitting],type=[vsPlayer],opposingPlayerId={opponent_id},season={CURR_SEASON},sportId=1)'
    params = {'personId': batter_id, 'hydrate':hydrate, 'sportId':1}
    r = statsapi.get('person',params)
    
    # Look up batting stats versus pitcher, if atBats_h2h == 0 return 
    # a dictionary of empty stats.
    stats_wanted = ["avg", "obp", "ops", "slg"]

    try: 
        batting_stats = r['people'][0]['stats'][1]['splits'][0]['stat']
    except KeyError:
        return OrderedDict({s: 0.0 for s in stats_wanted})
    
    # Only get rate stats vs pitcher
    filtered = {(k + "_h2h"):(float(v) if v != "-.--" and v != ".---" and v != "*.**" else 0.0)
                for k, v in batting_stats.items() 
                if type(v) == str  # We want the values to be strings because in the API, rate stats are stored as strings
                and k in stats_wanted} 
    
    # Making sure the keys are in the same order regardless of players entered
    filtered = OrderedDict(sorted(filtered.items()))
    
    return filtered

def batting_past_N_games(N: int, player_id: int, season=CURR_SEASON) -> OrderedDict:  
    """
    Returns a dictionary containing a limited amount of batting statistics 
    over the past N games for a specified player. One of the main data retrieval 
    functions.
    
    Parameters 
    -----–-----------
    N: int
        Specifies how many games back to look for batting statistics.
    
    player_id: int
        The 6-digit ID of a hitter, which can be fetched using 
        get_player_id_from_name('Hitter Name').
    """
    
    hydrate = f'stats(group=[hitting],type=[lastXGames],limit={N},season={CURR_SEASON}),currentTeam'
    params = {'personId': player_id, 'hydrate':hydrate}
    
    # Attempt to look up stats over the past N games, and if nothing comes
    # up, return a list of stats containing only 0.0. 
    try:
        r = statsapi.get('person',params)
        batting_stats = r['people'][0]['stats'][0]['splits'][0]['stat']
    except (ValueError, KeyError):
        return {k:v for k, v in (zip(range(5), [0.0]*5))}
    
    # Only get rate stats for past N games
    filtered = {k + "_p{}G".format(N):(float(v) if v != "-.--" and v != ".---" and v != "*.**" else 0.0)
                for k, v in batting_stats.items() 
                if type(v) == str 
                and k != 'stolenBasePercentage'
                or k == 'hits'} 
    
    # Preserving order across players
    filtered = OrderedDict(sorted(filtered.items()))
    
    return filtered

def pitching_past_N_games(N: int, player_id: int, season=CURR_SEASON) -> OrderedDict:
    """
    Returns a dictionary containing a limited amount of pitching statistics 
    over the past N games for a specified player. One of the main data retrieval 
    functions.
    
    Parameters 
    -----–-----------
    N: int
        Specifies how many games back to look for pitching statistics.
    
    player_id: int
        The 6-digit ID of a pitcher, which can be fetched using 
        get_player_id_from_name('Pitcher Name').
    """
    
    # Jose Abreu's (1B) name gets looked up if you pass in 
    # an empty string to statsapi.lookup_player().
    if player_id == 547989:
        return {k:v for k, v in (zip(range(15), [0.0]*15))}
    
    hydrate = f'stats(group=[pitching],type=[lastXGames],limit={N},season={CURR_SEASON}),currentTeam'
    params = {'personId': player_id, 'hydrate':hydrate}
    
    try:
        r = statsapi.get('person',params)
    except ValueError:  # The request fails if a pitcher is making their debut
        return {k:v for k, v in (zip(range(15), [0.0]*15))}
    
    pitching_stats = r['people'][0]['stats'][0]['splits'][0]['stat']
    
    # Only get rate stats for past N days
    empty_formats = ["-.--", ".---", "*.**"]
    filtered = {(k + "_p{}G".format(N)):(float(v) if v not in empty_formats else 0.0)
                for k, v in pitching_stats.items() 
                if type(v) == str} 
    
    # Preserving order across players
    filtered = OrderedDict(sorted(filtered.items()))
    
    return filtered

def check_pitcher_right_handed(pitcher_id: int) -> bool:
    """
    Returns a bool indicating whether a pitcher is right handed.

    Parameters 
    -----–-----------
    pitcher_id: int
        The 6-digit ID of a pitcher, which can be fetched using 
        get_player_id_from_name('Pitcher Name').        
    """
    try:
        params = {'personId': pitcher_id}
        r = statsapi.get('person',params)
        return r['people'][0]['pitchHand']['code'] == 'R'
    except IndexError:
        return True # Most pitchers are righties

def check_batter_right_handed(batter_id: int) -> bool:
    """
    Returns a bool indicating whether a hitter is right handed.

    Parameters 
    -----–-----------
    batter_id: int
        The 6-digit ID of a batter, which can be fetched using 
        get_player_id_from_name('Hitter Name').        
    """
    try:
        params = {'personId': batter_id}
        r = statsapi.get('person',params)
        return r['people'][0]['batSide']['code'] == 'R'
    except IndexError:
        return True # Most batters are righties

def check_pitcher_batter_opposite_hand(batter_id: int, pitcher_id: int) -> bool:
    """
    Returns a bool indicating whether a batter and pitcher 
    have opposite handedness.

    Parameters 
    -----–-----------
    batter_id: int
        The 6-digit ID of a batter, which can be fetched using 
        get_player_id_from_name('Hitter Name').      
        
    pitcher_id: int
        The 6-digit ID of a pitcher, which can be fetched using 
        get_player_id_from_name('Pitcher Name'). 
    """
    return check_pitcher_right_handed(pitcher_id) != check_batter_right_handed(batter_id)

def player_got_hit_in_game(player_id: int, game_id: int, home_or_away: str) -> bool:
    """
    This function generates labels for training data. Checks if a 
    player got a hit in a specified game. 

    Parameters 
    -----–-----------
    player: int
        The 6-digit ID of a batter, which can be fetched using 
        get_player_id_from_name('Hitter Name').      
        
    game_id: int
        The 6-digit ID for a game, can be fetched from statsapi.schedule().
    
    home_or_away: str
        Indicates whether the player was on the home team or the 
        away team for the specified game. Value is either "home" or "away". 
    """
    
    params = {'gamePk':game_id,
      'fields': 'gameData,teams,teamName,shortName,teamStats,batting,atBats,runs,hits,rbi,strikeOuts,baseOnBalls,leftOnBase,players,boxscoreName,liveData,boxscore,teams,players,id,fullName,batting,avg,ops,era,battingOrder,info,title,fieldList,note,label,value'}
    r = statsapi.get('game', params)
    player_stats = r['liveData']['boxscore']['teams'][home_or_away]['players'].get('ID' + str(player_id), False)
    if not player_stats: 
        return False 
    else:
        return player_stats['stats']['batting'].get('hits', 0) > 0

def convert_to_FL_format(name: str) -> str:
    """
    Takes the name of a player in Last, First format and converts
    it to First Last format. 

    Parameters 
    -----–-----------
    name: str
        The name of a player in Last, First format.  
    """
    last_first = name.split(",")
    last_first.reverse()
    last_first[0] = last_first[0].strip()
    return " ".join(last_first)

# Note on optimization: This looks somewhat messy, but it seems
# that appending with native Python lists and then converting
# to a Pandas DataFrame at the end seems to be the fastest way
# to go about this. 

def get_hitters_data(team_id: int, 
                     game_id: int,
                     gameday: str,
                     home_team: bool,
                     team_names_dict: dict, 
                     pitcher_data_dict: dict, 
                     prob_pitcher_dict: dict,
                     generate_train_data: bool, 
                     rows_list: list) -> None:
    """
    Gets all the necessary data for model training for a specific 
    team. Used in the process of data collection. 

    Parameters 
    -----–-----------
    team_id: int
        The ID for a team.

    game_id: int
        The ID for the game used to generate data. 

    gameday: str
        The date of the game used to generate data. 

    home_team: bool
        If getting data for the home team, you need the stats
        for the away probable pitcher. This variable simply 
        determines whether to get the stats for the home or 
        away probable pitcher. 

    team_names_dict: dict
        A dictionary with two keys: "home" and "away", each
        mapped to the team names for the home team and
        the away team, respectively. 
    
    pitcher_data_dict: dict
        A dictionary with two keys: "home" and "away", each
        mapped to the pitcher stats for the home pitcher and
        the away pitcher, respectively. 

    prob_pitcher_dict: dict
        A dictionary with two keys: "home" and "away", each
        mapped to the pitcher ID for the home pitcher and
        the away pitcher, respectively. 

    generate_train_data: bool
        Generates labels along with the training data if True.

    rows_list: list
        A list to append generated rows to. 
    """
    team_name = team_names_dict['home'] if home_team else team_names_dict['away']
    pitcher_p5G = pitcher_data_dict['away'] if home_team else pitcher_data_dict['home']
    probable_pitcher = prob_pitcher_dict['away'] if home_team else prob_pitcher_dict['home'] 
    home_or_away = 'home' if home_team else 'away'

    hitter_list = get_hitters_list(team_id)
    for player in hitter_list:
        player_id = get_player_id_from_name(player)
        try:
            new_row = list(get_current_season_stats(player, team_name, gameday).values())
            new_row += list(batting_past_N_games(7, player_id).values())
            new_row += list(batting_past_N_games(15, player_id).values())
            new_row += list(pitcher_p5G.values())
            new_row += list(get_h2h_vs_pitcher(player_id, probable_pitcher).values())
            new_row.append(check_pitcher_batter_opposite_hand(batter_id=player_id, 
                                                                  pitcher_id=probable_pitcher))
            if generate_train_data:
                new_row.append(player_got_hit_in_game(player_id, game_id, home_or_away))

            rows_list.append(new_row)

        except (ValueError, IndexError, KeyError):
            continue

################################################## FUNCTION TO GENERATE DATA #######################################################

def generate_hits_data(generate_train_data=True, generate_from_date=yesterday) -> None:
    """
    Main data retrieval function. Combines all other functions defined
    above and generates data either for training or testing. Produces
    a dataframe and writes it to a CSV, putting it in the data/player_stats
    directory like so:
    
        data/player_stats/season_{CURR_SEASON}/player_stats_08_20_2019.csv
    
    The date at the end of the file name changes depending on the value
    passed for generate_train_data.

    Parameters 
    -----–-----------
    generate_train_data: bool
        Indicates whether the function should generate training or test
        data. Simply changes which day's games to look at. 

    generate_from_date: str
        Can be used to generate data from games on a specific date. 
    """

    ###############################################################
    # 
    # Change GENERATE_TRAIN_DATA to False to generate 
    # data for today's games instead, which won't have 
    # labels included for whether or not the player
    # got a hit
    #
    GENERATE_TRAIN_DATA = generate_train_data
    #
    ################################################################

    print("Starting data retrieval")

    output_path = Path(f"data/player_stats/season_{CURR_SEASON}")
    output_path.mkdir(parents=True, exist_ok=True)

    # GENERATE_TRAIN_DATA changes the day to generate data from.
    # The day to generate data from is assigned to the variable 
    # "gameday". 
    gameday = yesterday if GENERATE_TRAIN_DATA else today

    # Some functionality required for using the generate_from_date argument
    # First, check if the date is entered in the correct format
    try:
        gameday_datetime_obj = datetime.datetime.strptime(generate_from_date, '%m/%d/%Y')
    except ValueError as e:
        raise Exception("Please enter the date to generate data from in MM/DD/YYYY format") from e

    # Then, just check if the requested date is in the future
    if gameday_datetime_obj.year > CURR_SEASON:
        raise ValueError(f'The {gameday_datetime_obj.year} season has not started yet')

    # Checks if the requested date is not yesterday or today.
    # If it's not yesterday or today, then change the gameday
    # variable to the requested date. Otherwise, just keep 
    # the gameday variable as is (defined from the 
    # generate_train_data argument). 
    if generate_from_date != yesterday and generate_from_date != today:
        gameday = datetime.datetime.strftime(gameday_datetime_obj, '%m/%d/%Y')

    # Last error check: if all the other checks pass, make an API
    # request to get the schedule for the requested day. If there
    # were no games that day, raise an exception. 
    games = statsapi.schedule(gameday)
    if len(games) == 0: 
        raise ValueError(f"No games occurred on the day {gameday}")

    rows_list = []

    blob_exists = check_gcloud_blob_exists(
        str(output_path / f"player_stats_{gameday.replace('/', '_')}.csv")
    )
    if GENERATE_TRAIN_DATA and (check_generated_data(gameday.replace("/", "_")) or blob_exists): 
        # If player stats for yesterday already exist without labels...
        # Just label the existing data
        print("Labeling yesterday's data")

        file_to_get = str(output_path / f"player_stats_{gameday.replace('/', '_')}.csv")
        if blob_exists:
            download_blob(source_blob_name=file_to_get, destination_file_name=file_to_get)

        stats_yest = pd.read_csv(file_to_get)
        with_labels = label_yesterdays_player_stats(stats_yest, games)
        with_labels.to_csv(file_to_get, index=False)

        print(f"Generated file data/player_stats/player_stats_{gameday.replace('/', '_')}.csv")
        upload_blob(source_file_name=file_to_get, destination_blob_name=file_to_get)

        return

    print(f"Generating data for {gameday}")
    for game_idx, game in tqdm(enumerate(games)):
        if game['status'] == 'Postponed': 
            print('Postponed game, skipping')
            continue

        game_id = game['game_id']

        away_id = game['away_id']
        home_id = game['home_id']

        away_team_name = game['away_name']
        home_team_name = game['home_name']
        team_names_dict = {"home": home_team_name, "away": away_team_name}

        away_prob_Pname = convert_to_FL_format(game['away_probable_pitcher'])
        home_prob_Pname = convert_to_FL_format(game['home_probable_pitcher'])

        away_probable_pitcher = get_player_id_from_name(away_prob_Pname)
        home_probable_pitcher = get_player_id_from_name(home_prob_Pname)
        prob_pitcher_dict = {"home": home_probable_pitcher, "away": away_probable_pitcher}

        away_pitcher_p5G = pitching_past_N_games(5, away_probable_pitcher)
        home_pitcher_p5G = pitching_past_N_games(5, home_probable_pitcher)
        pitcher_data_dict = {"home": home_pitcher_p5G, "away": away_pitcher_p5G}

        # Get data for home team
        get_hitters_data(home_id, 
                         game_id=game_id,
                         gameday=gameday,
                         home_team=True, 
                         team_names_dict=team_names_dict,
                         pitcher_data_dict=pitcher_data_dict,
                         prob_pitcher_dict=prob_pitcher_dict,
                         generate_train_data=GENERATE_TRAIN_DATA,
                         rows_list=rows_list)
        
        # Get data for away team
        get_hitters_data(away_id, 
                         game_id=game_id,
                         gameday=gameday,
                         home_team=False, 
                         team_names_dict=team_names_dict,
                         pitcher_data_dict=pitcher_data_dict,
                         prob_pitcher_dict=prob_pitcher_dict,
                         generate_train_data=GENERATE_TRAIN_DATA,
                         rows_list=rows_list)

        print(f"Retrieved game {game_idx+1}/{len(games)}")

        
    sample_hitter = get_player_id_from_name("Brandon Crawford")
    sample_pitcher = get_player_id_from_name("Jacob DeGrom")
    player_stats_columns = list(get_current_season_stats("Brandon Crawford", "San Francisco Giants", 
                                                         CURR_SEASON_END).keys())
    player_stats_columns += list(batting_past_N_games(7, sample_hitter).keys())
    player_stats_columns += list(batting_past_N_games(15, sample_hitter).keys())
    player_stats_columns += list(pitching_past_N_games(5, sample_pitcher).keys())
    player_stats_columns += list(get_h2h_vs_pitcher(sample_hitter, sample_pitcher).keys())

    if GENERATE_TRAIN_DATA:
        player_stats_columns += ['pitcher_hitter_opposite_hand', 'player_got_hit']
    else:
        player_stats_columns += ['pitcher_hitter_opposite_hand']

    player_stats_table = pd.DataFrame(data=rows_list, columns=player_stats_columns).replace("-.--", 0.0)\
                                                                                    .replace(".---", 0.0).replace("*.**", 0.0)
    player_stats_table.insert(loc=0, column='date', value=gameday)
    file_to_generate = output_path / f"player_stats_{gameday.replace('/', '_')}.csv"
    player_stats_table.to_csv(file_to_generate, index=False)
    print("Finished generating file: {}".format(file_to_generate))

    upload_blob(source_file_name=str(file_to_generate), destination_blob_name=str(file_to_generate))

    
def check_generated_data(date) -> bool:
    """
    This function just checks if testing data was previously 
    generated for a date (just by checking for a local file).
    """
    stats_yest = f"data/player_stats/player_stats_{date}.csv"
    return os.path.isfile(stats_yest)

def label_yesterdays_player_stats(yesterday_stats, games):
    """
    Used in data retrieval if (unlabeled) test data has already
    been generated for yesterday's games. This function creates
    a two-column DataFrame that can be merged with the already
    existing player_stats data from yesterday. 
    
    Parameters 
    -----–-----------
    yesterday_stats: pd.DataFrame
        Yesterday's player statistics, usually from 
        pd.read_csv(<yesterday_data_file>).

    games: dict
        A list of yesterday's games and associated data, usually
        from statsapi.schedule(...). 

    """
    players = []
    labels = []

    if 'player_got_hit' in yesterday_stats.columns:
        print("Yesterday's data is already labeled, returning yesterday's data")
        return yesterday_stats

    print("Finding yesterday's games...")
    for game in games:
        game_id = game['game_id']
        away_id = game['away_id']
        home_id = game['home_id']
        away_team_name = game['away_name']
        home_team_name = game['home_name']

        home_player_list = get_hitters_list(home_id)
        players += home_player_list
        away_player_list = get_hitters_list(away_id)
        players += away_player_list

        home_player_ids = [get_player_id_from_name(h) for h in home_player_list]
        away_player_ids = [get_player_id_from_name(a) for a in away_player_list]

        home_labels = [player_got_hit_in_game(hid, game_id, 'home') for hid in home_player_ids]
        labels += home_labels
        away_labels = [player_got_hit_in_game(aid, game_id, 'away') for aid in away_player_ids]
        labels += away_labels

    label_df = pd.DataFrame.from_dict({'Name': players, 'player_got_hit': labels})
    with_labels = yesterday_stats.merge(label_df, on='Name')
    return with_labels

def generate_yesterdays_results() -> None:
    """
    Generates tables to put on the Past Results page for project 
    website. Puts the tables in the data/past_results directory.  
    """
    
    pred_yest = pd.read_csv("data/predictions/predictions_{}.csv".format(yesterday.replace("/", "_")))
    stats_yest = pd.read_csv("data/player_stats/player_stats_{}.csv".format(yesterday.replace("/", "_")))
    
    past_results = stats_yest[stats_yest['Name'].isin(pred_yest['Name'])].loc[:, ['Name', 'player_got_hit']]
    past_results['player_got_hit'] = past_results['player_got_hit'].apply(lambda x: "Yes" if x == 1.0 else "No")
    past_results = past_results.append({'Name': 'Overall Accuracy', 
                                        'player_got_hit': sum(past_results['player_got_hit'] == 'Yes') / 10}, 
                                       ignore_index=True)
    
    past_results.to_csv("data/past_results/past_results_{}.csv".format(yesterday.replace("/", "_")), index=False)
    print("Results for {} generated".format(yesterday))