#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic("reset", "-f")
    p = display
except NameError:
    p = print

import warnings

warnings.simplefilter("ignore")

import pandas as pd

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

input_dir = "march-machine-learning-mania-2025"


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


# In[19]:


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

Y = (
    game_stats_1_2.groupby(["Season", "TeamID_1", "TeamID_2"])["Spread_1"]
    .mean()
    .reset_index()
)
print(f"Y                  {Y.shape[0]:,}")
p(Y)
print()

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

game_stats_o_d = pd.concat([game_stats_1_o, game_stats_1_o])
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

season_stats = game_stats_o_d.rename(columns={"TeamID_o": "TeamID"})
season_stats_g = season_stats.groupby(["Season", "TeamID"])
season_stats = season_stats_g[
    [c for c in season_stats if c.endswith("_o")]
    + [c for c in season_stats if c.endswith("_d")]
]
season_stats = season_stats.mean()
season_stats = season_stats.reset_index()
season_stats = season_stats.sort_values(["Season", "TeamID"])
season_stats = season_stats.reset_index(drop=True)
print(f"season_stats   {season_stats.shape[0]:,}")
p(season_stats)
print()


# In[ ]:
