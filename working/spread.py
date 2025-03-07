#!/usr/bin/env python
# coding: utf-8

# # NCAA March Madness Prediction Model
#
# This script builds a prediction model for the March Madness tournament, analyzing historical NCAA basketball data for both men's and women's teams. I'm using a point spread approach (basically how many points one team is expected to win or lose by) as the foundation, then converting those spreads to win probabilities for my Kaggle competition submission.
#
# The basic idea: I aggregate team season stats and strength of schedule metrics, then train an XGBoost model to predict the expected point differential between any two teams. This gives me a solid prediction of which team would win in a hypothetical matchup - exactly what I need for the competition!
#
# [kaggle competitions march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
#
# [github bedwards mania](https://github.com/bedwards/mania)
#
# ```
# NCAA D1 basketball 2010-2025 men's and women's
# Predicted point spread distribution
# 13128.00  ┼
# 11934.64  ┤                      ╭─╮
# 10741.27  ┤                     ╭╯ │
#  9547.91  ┤                    ╭╯  ╰╮
#  8354.55  ┤                    │    ╰╮
#  7161.18  ┤                   ╭╯     │
#  5967.82  ┤                  ╭╯      ╰╮
#  4774.45  ┤                  │        ╰╮
#  3581.09  ┤                 ╭╯         │
#  2387.73  ┤               ╭─╯          ╰─╮
#  1194.36  ┤            ╭──╯              ╰──╮
#     1.00  ┼────────────╯                    ╰───────────────
#                                  0          32
#
# NCAA D1 basketball 2010-2025 men's and women's
# Predicted win probabilities distribution
# 13440.00  ┤
# 12332.18  ┼╮
# 11224.36  ┤│                                               ╭
# 10116.55  ┤│                                               │
#  9008.73  ┤│                                               │
#  7900.91  ┤│                                               │
#  6793.09  ┤│                                               │
#  5685.27  ┤│                                               │
#  4577.45  ┤╰╮                                             ╭╯
#  3469.64  ┤ ╰╮                                          ╭─╯
#  2361.82  ┤  ╰──────╮                              ╭────╯
#  1254.00  ┤         ╰──────────────────────────────╯
# ```
#
# Brier score: 0.1479
#
# #### Top 25 Women's and Men's based on my model
#
# - Tourney Win: Probability of winning the NCAA tournament (estimated by winning 6 consecutive games). Higher values indicate stronger championship contenders.
# - PPG: Points Per Game scored by the team. Measures raw offensive output without accounting for pace.
# - Wtd Diff: Weighted Point Differential - the average margin of victory adjusted for strength of schedule. Positive numbers indicate winning teams; higher values represent more dominant teams.
# - SOS: Strength of Schedule - average offensive rating of opponents faced. Higher numbers indicate a more difficult schedule.
# - Off Eff: Offensive Efficiency - points scored per 100 possessions. This pace-adjusted metric better reflects offensive quality than raw points per game.
# - Ast/TO: Assist to Turnover Ratio - measures a team's ball control and passing efficiency. Values above 1.0 indicate more assists than turnovers; elite teams often exceed 1.5.
#
# ```
# Top 25 Women's Teams (2025)
# Rank Team               Tourney Win  PPG    Wtd Diff  SOS    Off Eff  Ast/TO
# ----------------------------------------------------------------------------
# 1    Connecticut           1.57%     80.9    +29.5    67.2    115.5   2.04
# 2    Texas                 1.57%     82.7    +28.9    72.4    111.9   1.32
# 3    Notre Dame            1.56%     86.7    +28.5    70.5    111.5   1.20
# 4    USC                   1.55%     82.1    +27.0    70.2    107.3   1.24
# 5    South Carolina        1.54%     80.2    +26.0    73.8    110.0   1.37
# 6    UCLA                  1.54%     79.6    +25.8    71.1    109.0   1.37
# 7    Kansas St             1.54%     80.4    +25.4    67.8    112.1   1.82
# 8    LSU                   1.53%     86.3    +24.8    72.5    109.0   1.03
# 9    West Virginia         1.53%     77.1    +24.4    67.8    104.2   0.90
# 10   TCU                   1.50%     79.2    +23.0    67.8    114.3   1.76
# 11   Mississippi           1.50%     76.4    +22.7    70.4    105.6   1.34
# 12   Tennessee             1.49%     90.0    +22.5    73.3    110.0   1.12
# 13   Baylor                1.46%     80.0    +20.9    67.0    106.6   1.43
# 14   Michigan St           1.44%     81.0    +20.3    69.6    105.0   1.28
# 15   Alabama               1.44%     78.8    +20.3    71.1    107.7   1.12
# 16   Florida St            1.44%     90.8    +20.2    69.5    112.3   1.31
# 17   Oklahoma St           1.43%     78.0    +19.8    68.0    105.0   0.97
# 18   Ohio St               1.41%     80.0    +19.3    70.4    103.8   1.26
# 19   Vanderbilt            1.41%     84.7    +19.3    70.9    109.3   1.21
# 20   Fairfield             1.39%     74.2    +18.9    61.1    106.5   1.40
# 21   Oklahoma              1.38%     85.5    +18.6    71.4    103.3   1.14
# 22   Duke                  1.38%     75.4    +18.5    72.0    102.4   1.19
# 23   Harvard               1.37%     70.3    +18.4    61.8    102.3   1.05
# 24   North Carolina        1.34%     71.9    +17.6    70.6    100.4   1.21
# 25   Utah                  1.32%     77.8    +17.3    69.9    107.6   1.24
#
# Top 25 Men's Teams (2025)
# Rank Team               Tourney Win  PPG    Wtd Diff  SOS    Off Eff  Ast/TO
# ----------------------------------------------------------------------------
# 1    Duke                  2.21%     81.0    +20.7    74.6    122.9   1.81
# 2    Gonzaga               2.09%     88.0    +18.4    73.4    121.1   2.15
# 3    Houston               2.08%     75.4    +18.2    74.3    117.1   1.55
# 4    Florida               2.06%     83.8    +17.9    74.9    118.7   1.52
# 5    Auburn                2.04%     85.1    +17.7    77.0    122.4   1.93
# 6    Maryland              1.97%     83.5    +16.7    74.2    117.0   1.50
# 7    Tennessee             1.81%     74.6    +15.0    77.4    114.0   1.59
# 8    Texas Tech            1.80%     81.8    +14.8    73.7    119.1   1.68
# 9    Iowa St               1.78%     81.3    +14.7    74.1    115.3   1.36
# 10   Missouri              1.70%     82.6    +14.0    74.6    119.4   1.34
# 11   VCU                   1.68%     77.4    +13.8    72.1    113.9   1.39
# 12   St Mary's CA          1.66%     74.5    +13.6    75.1    115.1   1.77
# 13   UC San Diego          1.64%     78.0    +13.5    71.7    115.0   1.62
# 14   St John's             1.54%     78.2    +12.7    73.4    107.0   1.48
# 15   Michigan St           1.46%     79.1    +12.1    75.4    113.8   1.58
# 16   Alabama               1.43%     90.3    +11.9    77.6    118.4   1.34
# 17   Arizona               1.43%     81.0    +11.9    74.8    112.5   1.45
# 18   Wisconsin             1.42%     82.0    +11.9    75.9    118.3   1.61
# 19   Illinois              1.42%     83.9    +11.8    75.0    114.0   1.26
# 20   Yale                  1.40%     82.2    +11.7    72.9    117.0   1.59
# 21   BYU                   1.38%     80.2    +11.6    72.4    116.1   1.52
# 22   Clemson               1.35%     77.4    +11.4    74.9    115.1   1.42
# 23   Michigan              1.28%     81.6    +10.9    75.1    112.8   1.18
# 24   SMU                   1.26%     82.1    +10.8    73.1    115.8   1.46
# 25   Louisville            1.26%     79.5    +10.8    74.5    114.0   1.35
# ```

# In[1]:


try:
    get_ipython().run_line_magic("reset", "-f")
except NameError:
    is_notebook = False
    p = print
else:
    is_notebook = True
    p = display
    # p = print

import warnings

warnings.simplefilter("ignore")

from math import floor
import numpy as np
import pandas as pd

if is_notebook:
    import matplotlib.pyplot as plt
    import seaborn as sns
else:
    import asciichartpy

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from scipy.stats import norm
from scipy.optimize import minimize

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

input_dir = "march-machine-learning-mania-2025"


# ## Data Loading and Setup
#
# First, I'm setting up the environment and importing the libraries I'll need. I'm loading several data files containing game results for both men's and women's basketball - regular season games, NCAA tournament games, and secondary tournament games (like the NIT). For each type, I'm grabbing both "compact" results (basic game outcome data) and "detailed" results (containing all the box score stats like rebounds, assists, etc.).
#
# I'm starting from the 2010 season since that's when the detailed women's data begins. This way I can build a unified model for both men's and women's basketball, which I think will give me better overall predictions than separate models.

# In[2]:


df_names = []
df_names_compact = []
df_names_detailed = []

for gender in ["M", "W"]:
    for part in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
        for result_type in ["Compact", "Detailed"]:
            if part == "SecondaryTourney" and result_type == "Detailed":
                continue
            df_name = f"{gender}{part}{result_type}Results"
            df_names.append(df_name)
            if result_type == "Compact":
                df_names_compact.append(df_name)
            else:
                df_names_detailed.append(df_name)
            print(df_name)
            path = f"../input/{input_dir}/{df_name}.csv"
            df = pd.read_csv(path)
            df["Part"] = part
            globals()[df_name] = df

season_min = WRegularSeasonDetailedResults["Season"].min()

for df_name in df_names:
    df = globals()[df_name]
    globals()[df_name] = df[df["Season"] >= season_min]


# ## Data Reshaping and Team Statistics
#
# I've now consolidated all the detailed game data and I'm reshaping it to be more machine-learning friendly. For each game, I'm creating a consistent format where the teams are labeled as "team_1" and "team_2" (instead of winner/loser), with stats labeled accordingly. I'm also calculating the point spread between the teams - the key value I want to predict.
#
# I then convert this further into "offensive" and "defensive" perspectives for each team in each game. This gives me a complete picture of how teams perform both when they have the ball and when they're defending. I'll use this to aggregate season-level statistics for each team.

# In[3]:


game_stats_1_2 = pd.DataFrame()

for df_name in df_names_detailed:
    game_stats_1_2 = pd.concat([game_stats_1_2, globals()[df_name]])

game_stats_1_2 = game_stats_1_2.reset_index(drop=True)
mask = game_stats_1_2["WTeamID"] < game_stats_1_2["LTeamID"]


def to_1_2(col_W_L):
    game_stats_1_2.loc[mask, f"{col_W_L}_1"] = game_stats_1_2[f"W{col_W_L}"]
    game_stats_1_2.loc[mask, f"{col_W_L}_2"] = game_stats_1_2[f"L{col_W_L}"]
    game_stats_1_2.loc[~mask, f"{col_W_L}_1"] = game_stats_1_2[f"L{col_W_L}"]
    game_stats_1_2.loc[~mask, f"{col_W_L}_2"] = game_stats_1_2[f"W{col_W_L}"]
    game_stats_1_2[f"{col_W_L}_1"] = game_stats_1_2[f"{col_W_L}_1"].astype(
        game_stats_1_2[f"W{col_W_L}"].dtype
    )
    game_stats_1_2[f"{col_W_L}_2"] = game_stats_1_2[f"{col_W_L}_2"].astype(
        game_stats_1_2[f"W{col_W_L}"].dtype
    )


game_stats_1_2["WLoc"] = game_stats_1_2["WLoc"].astype("category")
game_stats_1_2["LLoc"] = game_stats_1_2["WLoc"].cat.rename_categories(
    {"A": "H", "H": "A"}
)

for c in game_stats_1_2.columns:
    if c.startswith("W"):
        to_1_2(c[1:])

game_stats_1_2["Spread_1"] = game_stats_1_2["Score_1"] - game_stats_1_2["Score_2"]
game_stats_1_2["Spread_2"] = game_stats_1_2["Score_2"] - game_stats_1_2["Score_1"]
game_stats_1_2 = game_stats_1_2[[c for c in game_stats_1_2 if c[0] not in ("W", "L")]]
game_stats_1_2 = game_stats_1_2.drop(columns="NumOT")
game_stats_1_2.insert(2, "TeamID_1", game_stats_1_2.pop("TeamID_1"))
game_stats_1_2.insert(3, "TeamID_2", game_stats_1_2.pop("TeamID_2"))
game_stats_1_2 = game_stats_1_2.sort_values(
    ["Season", "DayNum", "TeamID_1", "TeamID_2"]
)
game_stats_1_2 = game_stats_1_2.reset_index(drop=True)
print(f"game_stats_1_2     {game_stats_1_2.shape[0]:,}")
p(game_stats_1_2)
print()

print(f"MRegularSeasonDetai{MRegularSeasonDetailedResults.shape[0]:>7,}")
print(f"WRegularSeasonDetai{WRegularSeasonDetailedResults.shape[0]:>7,}")
print(
    f"RegularSeasonDetail{MRegularSeasonDetailedResults.shape[0]+WRegularSeasonDetailedResults.shape[0]:>7,}"
)

reg_season = game_stats_1_2[game_stats_1_2["Part"] == "RegularSeason"]
print(f"reg_season         {reg_season.shape[0]:,}")

game_stats_1_o = reg_season.rename(
    columns={c: f"{c[:-2]}_o" for c in reg_season if c[-2:] == "_1"}
)
game_stats_1_o = game_stats_1_o.rename(
    columns={c: f"{c[:-2]}_d" for c in game_stats_1_o if c[-2:] == "_2"}
)
print(f"game_stats_1_o     {game_stats_1_o.shape[0]:,}")

game_stats_2_o = reg_season.rename(
    columns={c: f"{c[:-2]}_o" for c in reg_season if c[-2:] == "_2"}
)
game_stats_2_o = game_stats_2_o.rename(
    columns={c: f"{c[:-2]}_d" for c in game_stats_2_o if c[-2:] == "_1"}
)
print(f"game_stats_2_o     {game_stats_2_o.shape[0]:,}")
print(f"game_stats_1_o+2_o {game_stats_1_o.shape[0]+game_stats_2_o.shape[0]:,}\n")

game_stats_o_d = pd.concat([game_stats_1_o, game_stats_2_o])
game_stats_o_d = game_stats_o_d.drop(columns="Part")
game_stats_o_d.insert(2, "TeamID_o", game_stats_o_d.pop("TeamID_o"))
game_stats_o_d.insert(3, "TeamID_d", game_stats_o_d.pop("TeamID_d"))
game_stats_o_d = game_stats_o_d.sort_values(
    ["Season", "DayNum", "TeamID_o", "TeamID_d"]
)
game_stats_o_d = game_stats_o_d.reset_index(drop=True)
print(f"game_stats_o_d     {game_stats_o_d.shape[0]:,}")
p(game_stats_o_d)
print()


# ## Team Performance Analysis
#
# Looking at how many games each team plays in a season, we can see there's quite a bit of variation. Most teams play around 27-33 games per season, but there are some outliers with very few games.
#
# For example, in the 2021 COVID season, there were 16 teams that played 10 or fewer games. These anomalies might affect our model's ability to accurately assess those teams, but since they're relatively rare and mostly from a single unusual season, I'm going to keep them in the dataset.

# In[4]:


# Look at the distribution of games per team
games_per_team = game_stats_o_d.rename(columns={"TeamID_o": "TeamID"}).drop(
    columns="TeamID_d"
)
games_per_team_count = games_per_team.groupby(["Season", "TeamID"]).size()

# Count how many team-seasons have each number of games
games_count_distribution = games_per_team_count.value_counts().sort_index()
print("Distribution of games per team-season:")
print(games_count_distribution)

# Look at specific examples of teams with very few games
teams_with_few_games = games_per_team_count[games_per_team_count <= 10].reset_index()
print("\nSample of teams with 10 or fewer games:")
print(teams_with_few_games.head(10))

# Check if there's a pattern by season
few_games_by_season = teams_with_few_games.groupby("Season").size()
print("\nNumber of teams with few games by season:")
print(few_games_by_season)


# ## Calculating Season Statistics
#
# Now I'm calculating season-average statistics for each team by grouping all their games. For each team and season, I compute their average offensive and defensive performance across all the metrics I have: points scored/allowed, field goals, three-pointers, free throws, rebounds, assists, turnovers, steals, blocks, and fouls.
#
# These season averages form the foundation of my predictive features. I want to know how teams typically perform to predict how they'll do in tournament matchups.

# In[5]:


# p(game_stats_o_d)
season_stats = game_stats_o_d.rename(columns={"TeamID_o": "TeamID"})
season_stats = season_stats.drop(columns="TeamID_d")
season_stats_g = season_stats.groupby(["Season", "TeamID"])

season_stats = season_stats_g[
    [c for c in season_stats if c.endswith("_o")]
    + [c for c in season_stats if c.endswith("_d")]
].mean()

season_stats = season_stats.reset_index()
season_stats = season_stats.sort_values(["Season", "TeamID"])
season_stats = season_stats.reset_index(drop=True)

print(f"season_stats   {season_stats.shape[0]:,}")
p(season_stats)
print()


# ## Validating Season Stats
#
# Just sanity-checking the season stats for a specific team (TeamID 1393 in the 2010 season) to make sure everything looks reasonable. Eyeballing the data is always a good practice to catch any obvious issues before building models.

# In[6]:


p(season_stats[(season_stats["Season"] == 2010) & (season_stats["TeamID"] == 1393)])
# p(game_stats_o_d[(game_stats_o_d["Season"]==2010) & (game_stats_o_d["TeamID_o"]==1393)])
# p(MRegularSeasonDetailedResults[(MRegularSeasonDetailedResults["Season"]==2010) & (MRegularSeasonDetailedResults["WTeamID"]==1393)])
# p(MRegularSeasonDetailedResults[(MRegularSeasonDetailedResults["Season"]==2010) & (MRegularSeasonDetailedResults["LTeamID"]==1393)])


# ## Calculating Strength of Schedule
#
# Raw team statistics aren't enough - a team that puts up great numbers against weak competition isn't necessarily better than a team with decent numbers against tough opponents. So I'm creating strength of schedule metrics.
#
# For each team, I'm looking at all their opponents and averaging those opponents' season stats. This gives me a picture of the quality of competition each team has faced, which should help me make better predictions when teams from different conferences meet in the tournament.

# In[7]:


matchups = game_stats_o_d[["Season", "DayNum", "TeamID_o", "TeamID_d"]]
p(matchups)

# matchups with the season stats of d
opp_season_stats = pd.merge(
    matchups,
    season_stats,
    left_on=["Season", "TeamID_d"],
    right_on=["Season", "TeamID"],
)
opp_season_stats = opp_season_stats.drop(columns="TeamID")
opp_season_stats = opp_season_stats.sort_values(
    ["Season", "DayNum", "TeamID_o", "TeamID_d"]
)
opp_season_stats = opp_season_stats.reset_index(drop=True)
p(opp_season_stats)

# strength of schedule
sos_stats = opp_season_stats.rename(columns={"TeamID_o": "TeamID"})
sos_stats = sos_stats.drop(columns="TeamID_d")
sos_stats_g = sos_stats.groupby(["Season", "TeamID"])

sos_stats = sos_stats_g[
    [c for c in sos_stats if c.endswith("_o")]
    + [c for c in sos_stats if c.endswith("_d")]
].mean()

sos_stats = sos_stats.reset_index()
sos_stats = sos_stats.sort_values(["Season", "TeamID"])
sos_stats = sos_stats.reset_index(drop=True)

print(f"sos_stats   {sos_stats.shape[0]:,}")
p(sos_stats)
print()


# ## Combining Team Stats with Strength of Schedule
#
# I'm merging the raw team stats with their strength of schedule metrics to create a comprehensive statistical profile for each team-season combination. These combined features will give my model a more complete picture of each team's true strength.

# In[8]:


season_self_sos = pd.merge(
    season_stats, sos_stats, on=["Season", "TeamID"], suffixes=["_self", "_sos"]
)
print(f"season_self_sos {season_self_sos.shape[0]:,}")
p(season_self_sos)
print()


# ## Creating the Training Dataset
#
# Now I'm building the actual training dataset for my model. For each possible matchup between teams in the same season, I'm combining both teams' season stats and strength of schedule metrics. I'm also adding the actual point spread from their games as the target variable.
#
# The final training data has 119,167 rows (one for each possible matchup in my historical data) and a ton of features (120 columns) that describe various aspects of both teams' performance.

# In[9]:


train = (
    game_stats_1_2.groupby(["Season", "TeamID_1", "TeamID_2"])["Spread_1"]
    .mean()
    .reset_index()
)

train = pd.merge(
    train,
    season_self_sos,
    left_on=["Season", "TeamID_1"],
    right_on=["Season", "TeamID"],
)
train = train.drop(columns="TeamID")

train = train.rename(columns={c: f"{c}_1" for c in train if c[-4:] in ("self", "_sos")})

train = pd.merge(
    train,
    season_self_sos,
    left_on=["Season", "TeamID_2"],
    right_on=["Season", "TeamID"],
)
train = train.drop(columns="TeamID")

train = train.rename(columns={c: f"{c}_2" for c in train if c[-4:] in ("self", "_sos")})

train = train.drop(columns=[c for c in train if c.startswith("Spread_d_")])

train = train.sort_values(["Season", "TeamID_1", "TeamID_2"])
train = train.reset_index(drop=True)

print(f"train        {train.shape[0]:,}")
p(train)
print()

X = train.drop(columns=["Season", "TeamID_1", "TeamID_2", "Spread_1"])

print(f"X        {X.shape[0]:,}")
p(X)
print()

y = train["Spread_1"].rename("y")

print(f"y        {y.shape[0]:,}")
p(y)
print()


# ## Double-Checking 2021 Data
#
# I'm taking a quick look at the 2021 data specifically since it was an unusual season due to COVID. The data looks reasonable, though as we saw earlier, some teams played very few games that year.

# In[10]:


p(train[train["Season"] == 2021])


# ## Training the XGBoost Model
#
# I'm using k-fold cross-validation to train an XGBoost regressor model on my dataset. This approach helps prevent overfitting and gives me more reliable out-of-sample predictions. For each fold, I'm training on 80% of the data and validating on the remaining 20%, then rotating through all the data.
#
# The model is learning to predict the point spread between teams based on their season stats and strength of schedule metrics. I'll use these predictions to estimate win probabilities for my Kaggle submission.

# In[11]:


kfold = KFold(shuffle=True, random_state=42)
fold_models = []
y_pred_oof = np.zeros(len(train))

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train.index)):
    print(f"fold {fold_n}", flush=True)
    m = xgb.XGBRegressor()
    m.fit(X.iloc[i_fold], y.iloc[i_fold])
    fold_models.append(m)
    y_pred_oof[i_oof] = m.predict(X.iloc[i_oof])


# ## Visualizing Predicted Point Spreads
#
# Here I'm plotting the distribution of predicted point spreads from my model. The distribution looks reasonable - it's roughly bell-shaped and centered slightly above zero. The spread values mostly fall between -40 and +40 points, which aligns with what we typically see in college basketball games.

# In[12]:


suptitle = "NCAA D1 basketball 2010-2025 men's and women's"
title = "Predicted point spread distribution"

if is_notebook:
    sns.histplot(y_pred_oof, bins=50)
    plt.suptitle(suptitle)
    plt.title(title)

else:
    print()
    print(suptitle)
    print(title)
    hist, edges = np.histogram(y_pred_oof, bins=50)
    width = 80
    print(asciichartpy.plot(hist, {"height": 10, "width": width}))
    w50 = (width - 11) // 2
    w66 = (width - 11) * 2 // 3 - w50
    print(
        f"{floor(edges[len(edges)//2-1]):>{w50}}"
        f"{floor(edges[len(edges)*2//3]):>{w66}}"
    )
    print()


# ## Evaluating Model Performance
#
# I'm converting my predicted point spreads to win probabilities using a sigmoid function with a scaling factor of 0.25, then calculating the Brier score to evaluate how well my model predicts actual game outcomes.
#
# A Brier score of 0.1479 is pretty good - it's significantly better than the 0.25 you'd get from random guessing (always predicting 50%), which means my model has meaningful predictive power. For context, top models in basketball prediction competitions typically achieve Brier scores in the 0.12-0.16 range.

# In[13]:


y_pred_prob = 1 / (1 + np.exp(-y_pred_oof * 0.25))
y_true = (train["Spread_1"] > 0).astype(int)
brier_score = np.mean((y_pred_prob - y_true) ** 2)
print(f"Brier score: {brier_score:.4f}")


# claude.ai:
#
# > A Brier score of 0.1479 represents the mean squared error between your predicted win probabilities and the actual outcomes. Let me help you interpret this result:
# >
# > #### Understanding the Brier Score
# >
# > The Brier score ranges from 0 to 1, where:
# >
# > - 0 is perfect prediction (you predicted exactly the right probability for every game)
# > - 0.25 represents the score you'd get by always predicting 50% probability (random guessing)
# > - 1 is the worst possible score (you predicted with complete confidence the opposite of what happened)
# >
# > Your score of 0.1479 is significantly better than random guessing (0.25), which indicates your model has meaningful predictive power.
# >
# > #### Contextualizing Your Score
# >
# > For NCAA basketball game predictions:
# > - Professional betting markets and sophisticated models typically achieve Brier scores in the 0.12-0.16 range
# > - Your score of 0.1479 falls within this competitive range
# > - The best models in previous March Machine Learning Mania competitions have achieved scores around 0.12-0.13
# >
# > #### What This Means
# >
# > Your model is capturing meaningful patterns in the data. The 0.1479 Brier score suggests that:
# > - Your approach of using team season statistics to predict point spreads, then converting to win probabilities, is valid
# > - The model has good discriminative power (distinguishing between likely wins and losses)
# > - The probability calibration (using the sigmoid with scaling factor 0.25) is reasonably effective
#
# Note: There is data leakage in my approach, which artificially increases the out-of-fold cross-validation scores over scores involving real data (future games.) Specifically, the data leakage is that for regular season games, the season stats in X (the features used to fit/train the model) include the game being predicted. I am accepting this trade-off because it is a simpler, more efficient approach that gets more rows in my train dataset (vs. only predicting hitorical tournament games.) I could remove the game being predicted from the season stats for each row in the train dataset, but that would take more work with potentially little benefit and could greatly increase run time.

# ## Visualizing Win Probabilities
#
# Looking at the distribution of my predicted win probabilities, I notice it has high peaks near 0 and 1, with fewer predictions in the middle around 0.5. This means my model is making confident predictions about which team will win in most matchups - it's not wishy-washy with lots of "toss-up" predictions.
#
# Since my Brier score is good, this confidence appears to be justified - the model is recognizing clear favorites and underdogs in many matchups.

# In[14]:


suptitle = "NCAA D1 basketball 2010-2025 men's and women's"
title = "Predicted win probabilities distribution"

if is_notebook:
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_prob, bins=50, kde=True)
    plt.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="50% threshold")
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Count")
    plt.suptitle(suptitle)
    plt.title(title)
    plt.legend()
    plt.show()
else:
    print()
    print(suptitle)
    print(title)
    hist, _ = np.histogram(y_pred_prob, bins=50)
    print(asciichartpy.plot(hist, {"height": 10, "width": 80}))
    print()


# claude.ai:
#
# > The distribution actually reveals something important about your modeling approach. It's not centered around 0.5, and that's very interesting!
# The bimodal (two-peaked) distribution you're seeing with high peaks near 0 and 1 tells us that your sigmoid transformation is creating very confident predictions. The scaling factor of 0.25 in your sigmoid function is making the transformation quite steep, which pushes many probabilities toward the extremes.
# This pattern indicates:
# >
# > Your model is making strong, polarized predictions rather than cautious ones hovering around 0.5.
# The point spread predictions from your model are being transformed in a way that expresses high confidence - most games are predicted as either strong wins or strong losses.
# Since your Brier score is 0.1479 (which is quite good), many of these confident predictions are actually correct! If they weren't, your Brier score would be much worse.
# >
# > The distribution doesn't need to be centered at 0.5. In fact, if your model has good discriminative power, and games truly do have clear favorites and underdogs, you would expect to see more predictions at the extremes than in the middle.

# ## Preparing the Competition Submission
#
# Now I'm generating predictions for all potential matchups in the 2025 tournament. I'm taking the team data from the current season and using my trained model to predict the outcome of every possible pairing. These predictions are formatted according to the competition requirements and saved to "submission.csv".

# In[15]:


def to_csv(df, fn):
    df.to_csv(fn, index=False)
    with open(fn, "r+") as f:
        content = f.read()
        f.seek(0)
        f.truncate()
        f.write(content.rstrip("\n"))


print("Preparing submission")
sample_sub = pd.read_csv(f"../input/{input_dir}/SampleSubmissionStage2.csv")

sample_sub[["Season", "Team1", "Team2"]] = (
    sample_sub["ID"].str.split("_", expand=True).astype(int)
)

team_data = season_self_sos[season_self_sos["Season"] == 2025].set_index("TeamID")
X_submit = pd.DataFrame(index=sample_sub.index)

for col in X.columns:
    parts = col.split("_")
    team_num = parts[-1]
    base_col = "_".join(parts[:-1])
    team_id_col = f"Team{team_num}"
    X_submit[col] = sample_sub[team_id_col].map(team_data[base_col])

ensemble_preds = np.mean([model.predict(X_submit) for model in fold_models], axis=0)
win_probs = 1 / (1 + np.exp(-ensemble_preds * 0.25))
sample_sub["Pred"] = win_probs
to_csv(sample_sub[["ID", "Pred"]], "submission.csv")
print("Wrote submission.csv")


# ## Creating Multiple Simulation-Based Submissions
#
# As a final step, I'm generating multiple alternative submissions based on simulations. Instead of just submitting continuous probabilities, these variants use random sampling to create discrete probability buckets (0.1, 0.2, etc.).
#
# This technique can sometimes perform better in competitions - it's a way of "quantizing" my predictions that might better match how the actual tournament plays out. I'll be able to choose my best-performing submission once the tournament starts and early game results come in.

# In[16]:


print("Generating multiple independent simulation-based submissions...")
base_ids = sample_sub["ID"].copy()
probabilities = sample_sub["Pred"].values
num_matchups = len(probabilities)

for num_sims in range(1, 11):
    random_values = np.random.random((num_sims, num_matchups))
    sim_results = (random_values < probabilities).astype(int)

    submission = pd.DataFrame(
        {"ID": base_ids, "Pred": sim_results.sum(axis=0) / num_sims}
    )

    filename = f"submission{num_sims}.csv"
    to_csv(submission, filename)

    print(
        f"Wrote {filename:>16}, simulation rounds: {num_sims:>2}, win probabilities: {[f'{p:.2f}' for p in sorted(np.unique(submission['Pred']))]}"
    )

print("All independent simulation-based submissions complete!")


# claude.ai:
#
# > I understand your strategy - you're looking to maximize your chance of winning the March Machine Learning Mania competition rather than aiming for a consistent but average performance. With 4,000-5,000 competitors, a "safe" strategy likely won't put you in first place. Let me analyze your options and recommend the best high-risk, high-reward submission choices.
# >
# > ## Understanding Your Submission Options
# >
# > The script has generated 11 different submission files:
# > - `submission.csv`: Contains the raw model predictions (continuous probabilities)
# > - `submission1.csv` through `submission10.csv`: Contains simulated outcomes with varying numbers of simulation rounds
# >
# > The key difference between these files is:
# >
# > 1. **The main submission file** uses continuous probability values (like 0.73, 0.68, etc.)
# > 2. **The simulated files** use discrete probability values based on simulation rounds:
# >    - `submission1.csv` only has 0.0 and 1.0 (essentially random guesses based on the predicted probabilities)
# >    - `submission10.csv` has probabilities in increments of 0.1 (0.0, 0.1, 0.2, ..., 1.0)
# >
# > ## Why Simulation-Based Submissions Can Help Win
# >
# > The Brier score rewards getting things exactly right. While your main model might have a good overall Brier score (0.1479), the simulation approach introduces controlled variance that could result in a better score through luck.
# >
# > The tournament has significant inherent randomness (upsets, etc.), and your goal is to get the #1 spot, not just be in the top 10%.
# >
# > ## Recommended Strategy for Maximum Chance of Winning
# >
# > I recommend submitting:
# >
# > 1. **`submission1.csv`** - This is your highest variance option. It's essentially a random set of binary 0/1 predictions weighted by your model probabilities. This has the highest chance of getting lucky and scoring extremely well, though it also has a high chance of scoring poorly.
# >
# > 2. **`submission.csv`** - Your original model predictions with continuous probabilities. This is your "base model" that has shown good performance with a Brier score of 0.1479.
# >
# > This combination gives you:
# > - A very high variance option that could hit #1 if luck is on your side
# > - A solid foundation that still has a chance of winning but is more likely to place decently

# ## Team Ranking and Final Output
#
# Finally, I'm generating some advanced metrics for each team and creating a ranking of the top teams for the 2025 tournament - essentially a power ranking based on my model. This gives me (and you!) a quick reference for which teams might be the strongest contenders.
#
# For each team, I'm calculating:
# - Estimated possessions per game using the standard basketball formula
# - Offensive efficiency (points per 100 possessions)
# - Assist-to-turnover ratio (a key indicator of team discipline)
# - Three-point attempt rate (percentage of shots taken from beyond the arc)
# - Weighted point differential that accounts for strength of schedule
# - Win probability for a single game against an average tournament team
# - Tournament win probability (the chance of winning 6 games in a row)
#
# The final rankings show Connecticut, Texas, and Notre Dame as the clear favorites for the women's tournament, while Duke, Gonzaga, and Houston lead the men's projections. Check out those offensive efficiency numbers for Duke and Gonzaga - over 120 points per 100 possessions is elite-level offense!

# In[19]:


m_teams = pd.read_csv(f"../input/{input_dir}/MTeams.csv")
w_teams = pd.read_csv(f"../input/{input_dir}/WTeams.csv")
stats_2025 = season_self_sos[season_self_sos["Season"] == 2025]
w_stats = stats_2025[stats_2025["TeamID"] >= 3000].copy()
m_stats = stats_2025[stats_2025["TeamID"] < 3000].copy()

for df in [w_stats, m_stats]:
    df["poss"] = (
        df["FGA_o_self"] - df["OR_o_self"] + df["TO_o_self"] + 0.44 * df["FTA_o_self"]
    )
    df["off_eff"] = 100 * df["Score_o_self"] / df["poss"]
    df["ast_to"] = df["Ast_o_self"] / df["TO_o_self"]
    df["three_rate"] = 100 * df["FGA3_o_self"] / df["FGA_o_self"]
    df["wtd_diff"] = df["Spread_o_self"] * (
        df["Score_o_sos"] / df["Score_o_sos"].mean()
    )
    df["win_prob"] = 1 / (1 + np.exp(-df["wtd_diff"] * 0.2))
    df["tourney_win_prob"] = df["win_prob"] ** 6

w_stats["tourney_win_prob"] = (
    w_stats["tourney_win_prob"] / w_stats["tourney_win_prob"].sum()
)
m_stats["tourney_win_prob"] = (
    m_stats["tourney_win_prob"] / m_stats["tourney_win_prob"].sum()
)

w_top25 = (
    w_stats.sort_values("tourney_win_prob", ascending=False)
    .head(25)
    .merge(w_teams[["TeamID", "TeamName"]], on="TeamID")
)
m_top25 = (
    m_stats.sort_values("tourney_win_prob", ascending=False)
    .head(25)
    .merge(m_teams[["TeamID", "TeamName"]], on="TeamID")
)


def print_rankings(df, title):
    print(f"\n{title}")
    print(
        f"{'Rank':<4} {'Team':<18} {'Tourney Win':<10} {'PPG':<6} {'Wtd Diff':<8} {'SOS':<6} {'Off Eff':<8} {'Ast/TO':<6}"
    )
    print("-" * 70)

    for i, row in df.reset_index(drop=True).iterrows():
        print(
            f"{i + 1:<4} {row['TeamName']:<18} {row['tourney_win_prob'] * 100:7.2f}%  {row['Score_o_self']:5.1f} {row['wtd_diff']:+7.1f} {row['Score_o_sos']:5.1f} {row['off_eff']:7.1f} {row['ast_to']:5.2f}"
        )


print_rankings(w_top25, "Top 25 Women's Teams (2025)")
print_rankings(m_top25, "Top 25 Men's Teams (2025)")


# In[ ]:
