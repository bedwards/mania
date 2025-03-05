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

import os
from glob import glob
from itertools import combinations
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
input_dir = "march-machine-learning-mania-2025"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
df_names = []
df_names_with_id = []
df_names_with_season = []

for path in glob(f"../input/{input_dir}/*.csv"):
    df_name = path.split("/")[-1].split(".")[0]
    df_names.append(df_name)

    while df_name not in globals():
        try:
            df = pd.read_csv(path)
            if "ID" in df:
                df_names_with_id.append(df_name)
                df = df.set_index("ID").sort_index()
            if "Season" in df:
                df_names_with_season.append(df_name)
            globals()[df_name] = df

        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")
            df.to_csv(path, encoding="utf-8", index=False)

df_names.sort()
df_names_with_id.sort()
df_names_with_season.sort()

# for df_name in df_names:
#     df = globals()[df_name]
#     print(df_name)
#     p(df)
#     pring()

print(", ".join(df_names))
print()
print("ID:", ", ".join(df_names_with_id))
print()
print("Season:", ", ".join(df_names_with_season))


# In[2]:


print("SampleSubmissionStage1")
p(SampleSubmissionStage1)
print()
print("SeedBenchmarkStage1")
p(SeedBenchmarkStage1)
print()
print("SampleSubmissionStage2")
p(SampleSubmissionStage2)
print()
print("MTeams")
p(MTeams)
print()
print("WTeams")
p(WTeams)
print()
print("MTeamConferences")
p(MTeamConferences)
print()
print("WTeamConferences")
p(WTeamConferences)
print()


# In[3]:


path = f"{output_dir}/SubmissionIDAll.csv"

if glob(path):
    sub = pd.read_csv(path).set_index("ID")

else:
    sub = pd.DataFrame()

    for tc in [MTeamConferences, WTeamConferences]:
        for season in range(tc["Season"].min(), tc["Season"].max() + 1):
            tc_season = tc[tc["Season"] == season]
            assert tc_season.shape[0] == tc_season["TeamID"].nunique()
            sub_season = pd.DataFrame(
                [
                    (f"{season}_{t1}_{t2}", season, t1, t2)
                    for t1, t2 in combinations(tc_season["TeamID"], 2)
                ],
                columns=["ID", "Season", "TeamID1", "TeamID2"],
            )
            sub_season = pd.merge(
                sub_season,
                tc_season,
                left_on="TeamID1",
                right_on="TeamID",
                suffixes=("", "_1"),
            )
            sub_season = pd.merge(
                sub_season,
                tc_season,
                left_on="TeamID2",
                right_on="TeamID",
                suffixes=("", "_2"),
            )
            sub_season = sub_season.rename(
                columns={"TeamID": "TeamID_1", "ConfAbbrev": "ConfAbbrev_1"}
            )
            sub_season = sub_season.drop(
                columns=["TeamID1", "TeamID2", "Season_1", "Season_2"]
            )
            sub = pd.concat([sub, sub_season])

    sub = sub.set_index("ID").sort_index()
    sub.to_csv(path)
print("SubmissionIDAll (sub)")
p(sub)
print()


# In[4]:


path = f"{output_dir}/SubmissionStage1.csv"

if glob(path):
    SubmissionStage1 = pd.read_csv(path).set_index("ID")

else:
    SubmissionStage1 = sub[(sub["Season"] > 2020) & (sub["Season"] < 2025)]
    SubmissionStage1["Pred"] = 0.5
    SubmissionStage1 = SubmissionStage1[["Pred"]]
    SubmissionStage1.to_csv(path)

assert all(SampleSubmissionStage1 == SubmissionStage1)
print("SubmissionStage1")
p(SubmissionStage1)
print()


# In[5]:


for gender in ["M", "W"]:
    for result_type in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
        df_name = gender + result_type + "CompactResults"
        print(df_name)
        p(globals()[df_name])
        print()


# In[6]:


def wl_to_12(results, col):
    dtype = results[f"W{col}"].dtype
    mask = results["WTeamID"] < results["LTeamID"]
    results.loc[mask, f"{col}_1"] = results.loc[mask, f"W{col}"]
    results.loc[mask, f"{col}_2"] = results.loc[mask, f"L{col}"]
    results.loc[~mask, f"{col}_1"] = results.loc[~mask, f"L{col}"]
    results.loc[~mask, f"{col}_2"] = results.loc[~mask, f"W{col}"]
    results[f"{col}_1"] = results[f"{col}_1"].astype(dtype)
    results[f"{col}_2"] = results[f"{col}_2"].astype(dtype)
    return results, mask


def process_results(results):
    results, mask = wl_to_12(results, "TeamID")
    results["ID"] = (
        results["Season"].astype(str)
        + "_"
        + results["TeamID_1"].astype(str)
        + "_"
        + results["TeamID_2"].astype(str)
    )
    results["index"] = results["ID"] + "_" + results["DayNum"].astype(str)
    return results, mask


# In[7]:


Y_all = pd.DataFrame()

for gender in ["M", "W"]:
    for part in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
        results = globals()[gender + part + "CompactResults"].copy()
        results, mask = process_results(results)
        results["Part"] = part
        results["y_true"] = mask.astype(int)
        results = results[
            [
                "index",
                "ID",
                "DayNum",
                "Season",
                "Part",
                "TeamID_1",
                "TeamID_2",
                "WTeamID",
                "LTeamID",
                "y_true",
            ]
        ]
        Y_all = pd.concat([Y_all, results])

assert Y_all["index"].nunique() == Y_all.shape[0]
Y_all = Y_all.set_index("index").sort_index()
print("Y_all")
p(Y_all)
print()


# In[8]:


p(MRegularSeasonDetailedResults)
p(WRegularSeasonDetailedResults)


# In[9]:


GameStatsByID = pd.DataFrame()
cols_info = [
    "index",
    "ID",
    "Season",
    "TeamID_1",
    "TeamID_2",
    "DayNum",
    "NumOT",
    "Loc_1",
    "Loc_2",
]
cols_stats = [
    "Score",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]

for gender in ["M", "W"]:
    results = globals()[gender + "RegularSeasonDetailedResults"].copy()
    results, mask = process_results(results)

    for col in cols_stats:
        results, _ = wl_to_12(results, col)

    results["WLoc"] = results["WLoc"].astype("category")
    assert sorted(results["WLoc"].cat.categories.to_list()) == ["A", "H", "N"]
    results["LLoc"] = results["WLoc"].cat.rename_categories({"A": "H", "H": "A"})
    results, _ = wl_to_12(results, "Loc")
    cols = cols_info.copy()

    for col in cols_stats:
        for team in [1, 2]:
            cols.append(f"{col}_{team}")

    results = results[cols]
    GameStatsByID = pd.concat([GameStatsByID, results])

assert GameStatsByID["index"].nunique() == GameStatsByID.shape[0]
GameStatsByID = GameStatsByID.set_index("index").sort_index()
print("GameStatsByID")
p(GameStatsByID)
print()


# In[10]:


for part in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
    print(
        f"{part:>16} {Y_all[(Y_all['Season']>2002) & (Y_all['Part']==part)].shape[0]:>6}"
    )

print(f"{' '*16} {Y_all[Y_all['Season']>2002].shape[0]:>6}")


# In[11]:


df = GameStatsByID.reset_index().rename(columns={"index": "GameStatsByID_index"})
cols_single = ["GameStatsByID_index", "ID", "DayNum", "Season", "NumOT"]
cols_double = [
    "TeamID",
    "Loc",
    "Score",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]
GameStatsByTeam = pd.DataFrame(
    {c: df[[f"{c}_1", f"{c}_2"]].values.flatten() for c in cols_double}
)

for col in cols_single:
    GameStatsByTeam[col] = np.repeat(df[col].values, 2)

GameStatsByTeam["index"] = (
    GameStatsByTeam["GameStatsByID_index"] + "_" + np.tile(["1", "2"], len(df))
)

GameStatsByTeam = GameStatsByTeam[
    ["index"] + cols_single[:-1] + ["TeamID", "NumOT"] + cols_double[1:]
]
paired_indices = np.tile([1, 0], len(df))
game_indices = np.repeat(np.arange(len(df)), 2)

for col in cols_double[1:]:  # Skip TeamID
    GameStatsByTeam[f"{col}_o"] = GameStatsByTeam[col].values[
        game_indices * 2 + paired_indices
    ]

GameStatsByTeam = GameStatsByTeam.set_index("index").sort_index()
print("GameStatsByTeam")
p(GameStatsByTeam)
print()


# In[12]:


df = GameStatsByTeam.reset_index().rename(columns={"index": "GameStatsByTeam_index"})
groupby = ["Season", "TeamID"]
mean = [
    "NumOT",
    "Score",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
    "Score_o",
    "FGM_o",
    "FGA_o",
    "FGM3_o",
    "FGA3_o",
    "FTM_o",
    "FTA_o",
    "OR_o",
    "DR_o",
    "Ast_o",
    "TO_o",
    "Stl_o",
    "Blk_o",
    "PF_o",
]

SeasonStats = pd.concat(
    [
        pd.crosstab(
            [df["Season"], df["TeamID"]],
            [df["Loc"]],
            normalize="index",
        ),
        df.groupby(groupby)[mean].mean(),
    ],
    axis=1,
)

SeasonStats.sort_index()
print("SeasonStats")
p(SeasonStats)
print()


# In[13]:


s = GameStatsByTeam.loc[
    (GameStatsByTeam["Season"] == 2003) & (GameStatsByTeam["TeamID"] == 1103), "NumOT"
]
print(sorted(s.to_list()))
print(s.sum(), s.count(), f"{s.mean():.4f}")


# In[14]:


min_year_women_detailed = SeasonStats[
    SeasonStats.index.get_level_values("TeamID") > 2999
].index.min()[0]
print(min_year_women_detailed)
cols_shared = ["Season", "ID", "DayNum", "Part", "y_true"]
Y_teams = []

for suffix in ["1", "2"]:
    Y_team = Y_all.rename(columns={f"TeamID_{suffix}": "TeamID"})
    Y_team = Y_team[Y_team["Season"] >= min_year_women_detailed]
    Y_team = Y_team.reset_index().set_index(["Season", "TeamID"]).sort_index()

    if suffix == "1":
        print(
            f"{'Y_team':>11} {Y_team.shape} {Y_team[Y_team.index.get_level_values('TeamID')>2999].index.min()}"
        )
        print(
            f"SeasonStats {str(SeasonStats.shape):>11} {SeasonStats[SeasonStats.index.get_level_values('TeamID')>2999].index.min()}"
        )

    Y_team = Y_team.join(SeasonStats).reset_index()
    Y_team = Y_team.set_index("index").sort_index()
    Y_team = Y_team.drop(columns=["WTeamID", "LTeamID"])

    if suffix == "1":
        Y_team = Y_team.drop(columns=["TeamID_2"])
    else:
        Y_team = Y_team.drop(columns=["TeamID_1"] + cols_shared)

    Y_team = Y_team.rename(
        columns={col: f"{col}_{suffix}" for col in Y_team if col not in cols_shared}
    )
    Y_teams.append(Y_team)

train = pd.concat([Y_teams[0], Y_teams[1]], axis=1)
train = train[
    cols_shared
    + ["TeamID_1", "TeamID_2"]
    + [
        c
        for c in train
        if not c.startswith("TeamID_") and (c.endswith("_1") or c.endswith("_2"))
    ]
]
print("train")
p(train)
print()


# In[45]:


kfold = KFold(shuffle=True, random_state=42)
m = xgb.XGBClassifier(objective="binary:logistic")
y_pred_oof = np.zeros(len(train))

t = train.reset_index()

for season in sorted(t["Season"].unique()):
    print(season)
    t_season = t[t["Season"] == season]
    X = t_season.drop(columns=["index", "Season", "ID", "Part", "y_true"])
    y = t_season["y_true"]
    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(t_season.index)):
        print(f"  {fold_n}")
        m.fit(X.iloc[i_fold], y.iloc[i_fold])
        y_pred = m.predict_proba(X.iloc[i_oof])[:, 1]
        y_pred_oof[t_season.iloc[i_oof].index] = y_pred
        if season == min_year_women_detailed and fold_n == 0:
            sns.histplot(y_pred)
            plt.show()
    print()

sns.histplot(y_pred_oof)


# In[30]:


# In[ ]:
