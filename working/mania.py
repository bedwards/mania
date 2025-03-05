#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic("reset", "-f")
except NameError:
    pass

import warnings

warnings.simplefilter("ignore")

import os
from glob import glob
from itertools import combinations
import numpy as np
import pandas as pd

pd.set_option("display.expand_frame_repr", False)
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
#     print(df)
#     print()

print(", ".join(df_names))
print()
print("ID:", ", ".join(df_names_with_id))
print()
print("Season:", ", ".join(df_names_with_season))


# In[2]:


print("SampleSubmissionStage1\n", SampleSubmissionStage1, "\n")
print("SeedBenchmarkStage1\n", SeedBenchmarkStage1, "\n")
print("SampleSubmissionStage2\n", SampleSubmissionStage2, "\n")
print("MTeams\n", MTeams, "\n")
print("WTeams\n", WTeams, "\n")
print("MTeamConferences\n", MTeamConferences, "\n")
print("WTeamConferences\n", WTeamConferences, "\n")


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

print(sub)


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
print(SubmissionStage1)


# In[11]:


for gender in ["M", "W"]:
    for result_type in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
        df_name = gender + result_type + "CompactResults"
        print(df_name)
        print(globals()[df_name])


# In[23]:


Y = pd.DataFrame()

for gender in ["M", "W"]:
    for result_type in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
        results = globals()[gender + result_type + "CompactResults"]
        mask = results["WTeamID"] < results["LTeamID"]
        results.loc[mask, "TeamID_1"] = results.loc[mask, "WTeamID"]
        results.loc[mask, "TeamID_2"] = results.loc[mask, "LTeamID"]
        results.loc[~mask, "TeamID_1"] = results.loc[~mask, "LTeamID"]
        results.loc[~mask, "TeamID_2"] = results.loc[~mask, "WTeamID"]
        results["TeamID_1"] = results["TeamID_1"].astype(int)
        results["TeamID_2"] = results["TeamID_2"].astype(int)
        results["ID"] = (
            results["Season"].astype(str)
            + "_"
            + results["TeamID_1"].astype(str)
            + "_"
            + results["TeamID_2"].astype(str)
        )
        results["y_true"] = (results["WTeamID"] < results["LTeamID"]).astype(int)
        results = results[
            ["ID", "Season", "TeamID_1", "TeamID_2", "WTeamID", "LTeamID", "y_true"]
        ]
        Y = pd.concat([Y, results])

Y = Y.set_index("ID").sort_index()
print(Y)


# In[ ]:
