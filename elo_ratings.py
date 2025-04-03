import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# Creates initial ratings for all teams in dataframe of match results
def create_team_elo(df):
    teams = list(np.unique(df['winning_team_name']))
    elo = pd.DataFrame(columns=['team', 'rating'])
    elo['team'] = teams
    elo['rating'] = [1000.0] * len(teams)
    elo['matches'] = [0] * len(teams)
    return elo


def calc_team_elo(df, elo):
    # Initialize team_tracker
    team_tracker = []

    # Precompute unique team names and their indices in the Elo dataframe
    team_indices = {team: elo.index[elo['team'] == team][0] for team in elo['team']}

    for i in range(len(df)):
        row = df.iloc[i]

        # Get pre-match Elo ratings
        winning_team_index = team_indices[row['winning_team_name']]
        losing_team_index = team_indices[row['losing_team_name']]
        winning_team_prematch_elo = elo.at[winning_team_index, 'rating']
        losing_team_prematch_elo = elo.at[losing_team_index, 'rating']

        # Calculate pre-match Elo difference and win percentage
        prematch_elo_difference = losing_team_prematch_elo - winning_team_prematch_elo
        winning_team_win_percentage = round(1 / (1 + (10 ** (prematch_elo_difference / 400))), 2)

        # Calculate Elo change
        elo_change = round((25 + 25 * row['NRR']) * (1 - winning_team_win_percentage), 2)

        # Calculate post-match Elo ratings
        winning_team_postmatch_elo = winning_team_prematch_elo + elo_change
        losing_team_postmatch_elo = losing_team_prematch_elo - elo_change

        # Update dataframe with calculated values
        df.at[i, 'winning_team_prematch_elo'] = winning_team_prematch_elo
        df.at[i, 'losing_team_prematch_elo'] = losing_team_prematch_elo
        df.at[i, 'prematch_elo_difference'] = prematch_elo_difference
        df.at[i, 'winning_team_win_percentage'] = winning_team_win_percentage
        df.at[i, 'elo_change'] = elo_change
        df.at[i, 'winning_team_postmatch_elo'] = winning_team_postmatch_elo
        df.at[i, 'losing_team_postmatch_elo'] = losing_team_postmatch_elo

        # Update Elo dataframes
        elo.at[winning_team_index, 'rating'] = winning_team_postmatch_elo
        elo.at[winning_team_index, 'matches'] += 1
        elo.at[losing_team_index, 'rating'] = losing_team_postmatch_elo
        elo.at[losing_team_index, 'matches'] += 1

        # Track team data for plotting
        team_tracker.append({'team': row['winning_team_name'], 'rating': winning_team_postmatch_elo,
                             'matches': elo.at[winning_team_index, 'matches']})
        team_tracker.append({'team': row['losing_team_name'], 'rating': losing_team_postmatch_elo,
                             'matches': elo.at[losing_team_index, 'matches']})

    # Create Elo tracker dataframe
    elo_tracker = pd.DataFrame(team_tracker)

    return df, elo, elo_tracker


def plot_team_elo_tracker(elo_tracker):
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df1 in elo_tracker.groupby('team'):
        df1.plot(x='matches', y='rating', ax=ax, label=label)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return fig, ax


def load_dls_table(filepath="dls_tables//dls_t20.csv", plot=False):
    dls_df = pd.read_csv(filepath)
    dls_df.head()

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(0, 10):
            dls_df.plot(x='overs', y=str(i), ax=ax)

    return dls_df


def project_score(over, ball, w, curr_score, dls_df):
    overs_rem = 20 - over
    per_ball = (dls_df[dls_df['overs'] == overs_rem][str(w)].values[0] -
                dls_df[dls_df['overs'] == overs_rem - 1][str(w)].values[0]) / 6
    resource_left = (dls_df[dls_df['overs'] == overs_rem][str(w)].values[0]) - (per_ball * (ball * 10))
    resource_used = 100 - resource_left
    runs_per_resource = curr_score / resource_used
    projection = curr_score + (runs_per_resource * resource_left)
    return np.round(projection, 1)


def expected_runs(over, w, par_score, dls_df):
    overs_rem = 20 - over
    per_ball = (dls_df[dls_df['overs'] == overs_rem][str(w)].values[0] -
                dls_df[dls_df['overs'] == overs_rem - 1][str(w)].values[0]) / 6
    runs_per_resource = par_score / 100
    exp_runs = runs_per_resource * per_ball
    return np.round(exp_runs, 1)


def wicket_value(over, w, par_score, dls_df):
    overs_rem = 20 - over
    resource_change = (dls_df[dls_df['overs'] == overs_rem][str(w)].values[0] -
                       dls_df[dls_df['overs'] == overs_rem][str(w + 1)].values[0])
    runs_per_resource = par_score / 100
    del_runs = runs_per_resource * resource_change
    return np.round(del_runs, 1)


# NB roles must be either 'bowler' or 'batter'
def create_player_elo(df, role="bowler"):
    roles = list(np.unique(df[role + '_scorecard_name']))
    player_elo = pd.DataFrame(columns=[role, 'rating'])
    player_elo[role] = roles
    player_elo['rating'] = [1000.0] * len(roles)
    player_elo['deliveries'] = [0] * len(roles)
    player_elo_tracker = pd.DataFrame(columns=[role, 'rating'])
    player_elo_tracker[role] = roles
    player_elo_tracker['rating'] = [1500.0] * len(roles)
    player_elo_tracker['deliveries'] = [0] * len(roles)
    return player_elo


# Load in the bbb csv files from cricsheet or PCS
def import_bbb_csv_file(filepath):
    # Read the CSV file and rename columns
    df = pd.read_csv(filepath, low_memory=False).rename(
        columns={'Match': 'match_id', 'Date': 'start_date',
                 'Start Time': 'start_time',
                 'Innings': 'innings_number', 'innings': 'innings_number',
                 'Over': 'over_no', 'Ball': 'ball_no',
                 'Batter': 'batter_scorecard_name', 'Bowler': 'bowler_scorecard_name',
                 'striker': 'batter_scorecard_name', 'bowler': 'bowler_scorecard_name',
                 'Batting Team': 'batting_team_name', 'Bowling Team': 'bowling_team_name',
                 'batting_team': 'batting_team_name', 'bowling_team': 'bowling_team_name',
                 'wicket_type': 'wicket', 'Wicket': 'wicket',
                 'Runs': 'runs', 'runs_off_bat': 'runs'}
    )
    print("Importing file...")

    # Create innings_id column
    df['innings_id'] = df['match_id'].astype(str) + " - " + df['innings_number'].astype(str)
    df['start_date'] = pd.to_datetime(df['start_date'], format='mixed', dayfirst=True)

    # Check if 'ball' column exists and reformat it
    if 'ball' in df.columns:
        # Use vectorized operations to split 'ball' into 'over_no' and 'ball_no'
        df['over_no'], df['ball_no'] = zip(*df['ball'].apply(lambda x: divmod(x, 1)))
        df['over_no'] = df['over_no'].astype(int)
        df['ball_no'] = (df['ball_no'] * 10).round().astype(int)
    else:
        # Adjust 'over_no' if 'ball' column does not exist
        df['over_no'] = df['over_no'] - 1

    # Sort the dataframe by date, then match, then delivery number
    df = df.sort_values(['start_date', 'match_id', 'innings_number', 'over_no', 'ball_no'], ignore_index=True)
    print("Done!")
    return df


# converts innings-by-innings results into match level data for team elo ratings
def import_statsguru_innings(filepath):
    # Read the CSV file
    inn_df = pd.read_csv(filepath)

    # Create match_id column using vectorized operations
    inn_df['match_id'] = inn_df.apply(lambda row: ' vs '.join(sorted([row['Team'], row['Opposition'].split('v ')[1]]))
                                                  + ' - ' + row['Start Date'], axis=1)

    # Initialize the dataframe with the required columns
    df = pd.DataFrame(columns=['match_id', 'start_date', 'ground_loc', 'winning_team_name', 'losing_team_name',
                               'winning_innings_number', 'winning_overs', 'winning_runs', 'winning_wickets',
                               'winning_RR', 'losing_innings_number', 'losing_overs', 'losing_runs', 'losing_wickets',
                               'losing_RR', 'NRR'])

    # Get unique match IDs
    matches = inn_df['match_id'].unique()

    # Process each match using vectorized operations
    for i, match in enumerate(matches):
        inn_rows = inn_df[inn_df['match_id'] == match]
        if 'won' in inn_rows['Result'].values:
            win_row = inn_rows[inn_rows['Result'] == 'won'].iloc[0]
            lose_row = inn_rows[inn_rows['Result'] == 'lost'].iloc[0]

            # Populate the dataframe with match details
            df.at[i, 'match_id'] = match
            df.at[i, 'start_date'] = pd.to_datetime(win_row['Start Date'])
            df.at[i, 'ground_loc'] = win_row['Ground']
            df.at[i, 'winning_team_name'] = win_row['Team']
            df.at[i, 'winning_innings_number'] = win_row['Inns']
            df.at[i, 'winning_overs'] = win_row['Overs']
            df.at[i, 'winning_runs'], df.at[i, 'winning_wickets'] = \
                (win_row['Score'].split('/') if '/' in win_row['Score'] else (win_row['Score'], 10))
            df.at[i, 'winning_RR'] = win_row['RPO']
            df.at[i, 'losing_team_name'] = lose_row['Team']
            df.at[i, 'losing_innings_number'] = lose_row['Inns']
            df.at[i, 'losing_overs'] = lose_row['Overs']
            df.at[i, 'losing_runs'], df.at[i, 'losing_wickets'] = \
                (lose_row['Score'].split('/') if '/' in lose_row['Score'] else (lose_row['Score'], 10))
            df.at[i, 'losing_RR'] = lose_row['RPO']
            df.at[i, 'NRR'] = float(df.at[i, 'winning_RR']) - float(df.at[i, 'losing_RR'])

    # Sort the dataframe by start_date and match_id
    df.sort_values(by=['start_date', 'match_id'], ignore_index=True, inplace=True)

    # Initialize additional columns for Elo ratings
    df['winning_team_prematch_elo'] = pd.Series(index=df.index)
    df['losing_team_prematch_elo'] = pd.Series(index=df.index)
    df['prematch_elo_difference'] = pd.Series(index=df.index)
    df['winning_team_win_percentage'] = pd.Series(index=df.index)
    df['elo_change'] = pd.Series(index=df.index)
    df['winning_team_postmatch_elo'] = pd.Series(index=df.index)
    df['losing_team_postmatch_elo'] = pd.Series(index=df.index)

    return df


def calc_player_elo(df, dls_df, bowler_elo, batter_elo, par_score=173.6):
    # Initialize columns
    df['expected_runs'] = pd.Series(index=df.index)
    df['bowler_pre_ball_elo'] = pd.Series(index=df.index)
    df['batter_pre_ball_elo'] = pd.Series(index=df.index)
    df['pre_ball_elo_difference'] = pd.Series(index=df.index)
    df['er_adjustment_factor'] = pd.Series(index=df.index)
    df['elo_change'] = pd.Series(index=df.index)
    df['bowler_post_ball_elo'] = pd.Series(index=df.index)
    df['batter_post_ball_elo'] = pd.Series(index=df.index)
    df['team_wickets'] = pd.Series(index=df.index)
    df['wicket_value'] = pd.Series(index=df.index)

    curr_inn_id = None
    curr_inn_wickets = 0
    batter_tracker = []
    bowler_tracker = []

    # Precompute DLS values
    dls_values = dls_df.set_index('overs').to_dict()
    max_overs = np.max(dls_df['overs'])

    # Loop through each delivery in order
    for i in tqdm(range(len(df)), desc="Processing deliveries", total=len(df), unit="ball"):
        row = df.iloc[i]
        wicket_flag = 0

        if curr_inn_id != row['innings_id']:
            curr_inn_id = row['innings_id']
            curr_inn_wickets = 0

        # Get pre-ball Elo ratings
        bowler_pre_ball_elo = bowler_elo.at[
            bowler_elo.index[bowler_elo['bowler'] == row['bowler_scorecard_name']][0], 'rating']
        batter_pre_ball_elo = batter_elo.at[
            batter_elo.index[batter_elo['batter'] == row['batter_scorecard_name']][0], 'rating']

        # Calculate pre-ball Elo difference and adjustment factor
        pre_ball_elo_difference = batter_pre_ball_elo - bowler_pre_ball_elo
        er_adjustment_factor = round((1 / (1 + (10 ** (pre_ball_elo_difference / 400)))) * 2, 2)

        # Calculate expected runs and wicket value
        overs_rem = max_overs - row['over_no']
        expected_runs = round(((dls_values[str(curr_inn_wickets)][overs_rem] - dls_values[str(curr_inn_wickets)][
            overs_rem - 1]) / 6) * (par_score/100), 1) if max_overs <= 50 else 0.4
        wicket_value = round((dls_values[str(curr_inn_wickets)][overs_rem] - dls_values[str(curr_inn_wickets + 1)][
            overs_rem]) * (par_score / 100), 1)

        # Update wickets if a wicket falls
        if not pd.isnull(row['wicket']):
            curr_inn_wickets += 1
            if curr_inn_wickets == 10:
                curr_inn_wickets = 9
            wicket_flag = 1

        # Calculate runs added and Elo change
        runs_added = (row['runs'] - expected_runs) - (wicket_flag * wicket_value)
        elo_change = round(runs_added * (er_adjustment_factor if runs_added >= 0 else (2 - er_adjustment_factor)), 1)

        # Calculate post-ball Elo ratings
        bowler_post_ball_elo = bowler_pre_ball_elo - elo_change
        batter_post_ball_elo = batter_pre_ball_elo + elo_change

        # Update dataframe with calculated values
        df.at[i, 'bowler_pre_ball_elo'] = bowler_pre_ball_elo
        df.at[i, 'batter_pre_ball_elo'] = batter_pre_ball_elo
        df.at[i, 'pre_ball_elo_difference'] = pre_ball_elo_difference
        df.at[i, 'er_adjustment_factor'] = er_adjustment_factor
        df.at[i, 'expected_runs'] = expected_runs
        df.at[i, 'wicket_value'] = wicket_value
        df.at[i, 'team_wickets'] = curr_inn_wickets
        df.at[i, 'runs_added'] = runs_added
        df.at[i, 'elo_change'] = elo_change
        df.at[i, 'bowler_post_ball_elo'] = bowler_post_ball_elo
        df.at[i, 'batter_post_ball_elo'] = batter_post_ball_elo

        # Update Elo dataframes
        bowler_elo.at[
            bowler_elo.index[bowler_elo['bowler'] == row['bowler_scorecard_name']][0], 'rating'] = bowler_post_ball_elo
        bowler_elo.at[bowler_elo.index[bowler_elo['bowler'] == row['bowler_scorecard_name']][0], 'deliveries'] += 1
        batter_elo.at[
            batter_elo.index[batter_elo['batter'] == row['batter_scorecard_name']][0], 'rating'] = batter_post_ball_elo
        batter_elo.at[batter_elo.index[batter_elo['batter'] == row['batter_scorecard_name']][0], 'deliveries'] += 1

        # Track player data for plotting
        bowler_tracker.append(
            {'bowler': row['bowler_scorecard_name'], 'team': row['bowling_team_name'], 'rating': bowler_post_ball_elo,
             'deliveries': bowler_elo.at[
                 bowler_elo.index[bowler_elo['bowler'] == row['bowler_scorecard_name']][0], 'deliveries']})
        batter_tracker.append(
            {'batter': row['batter_scorecard_name'], 'team': row['batting_team_name'], 'rating': batter_post_ball_elo,
             'deliveries': batter_elo.at[
                 batter_elo.index[batter_elo['batter'] == row['batter_scorecard_name']][0], 'deliveries']})

    # Create Elo tracker dataframes
    bowler_elo_tracker = pd.DataFrame(bowler_tracker)
    batter_elo_tracker = pd.DataFrame(batter_tracker)

    return df, bowler_elo, bowler_elo_tracker, batter_elo, batter_elo_tracker


def plot_top_players(elo_df, elo_tracker_df, role="bowler", n=10, thresh=100):
    # sort players by top ratings
    best = elo_df[elo_df['deliveries'] > thresh].sort_values('rating', ascending=False, ignore_index=True)

    # find top n players' names
    top_n = best[0:n][role]
    best_tracker = elo_tracker_df[elo_tracker_df[role].isin(list(top_n))]

    # plot delivery number vs rating
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df1 in best_tracker.groupby(role):
        df1.plot(x='deliveries', y='rating', ax=ax, label=label + ' - ' + str(round(df1['rating'].iloc[-1])))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return top_n
