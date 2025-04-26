import pandas as pd
import zipfile
import numpy as np
import os


# setup initial variables

# path to data
datapath = "Soccer Analytics (FS 2025) Data/"
matches_csv = datapath + "matches.csv"
wyscout_directory = datapath + "wyscout/"
skillcorner_directory = datapath + "skillcorner/"

# name of your country
country = "Sweden"



# load the correct ids
wyscout = []
skillcorner = []

if not os.path.exists(datapath):
    # unzip Soccer Analytics (FS 2025) Data.zip
    with zipfile.ZipFile("Soccer Analytics (FS 2025) Data.zip", "r") as zip_ref:
        zip_ref.extractall()

    # # untar Soccer Analytics (FS 2025) Data.tar
    # with tarfile.open("Soccer Analytics (FS 2025) Data.tar") as tar:
    #     tar.extractall()
else:
    # remove datapath
    shutil.rmtree(datapath)
    with zipfile.ZipFile("Soccer Analytics (FS 2025) Data.zip", "r") as zip_ref:
        zip_ref.extractall()
    # with tarfile.open("Soccer Analytics (FS 2025) Data.tar") as tar:
    #     tar.extractall()


df_matches = pd.read_csv(matches_csv)
df_matches_my_country = df_matches[(df_matches["home"] == country) | (df_matches["away"] == country)]
df_matches_my_country = df_matches_my_country.reset_index(drop=True)

countries = pd.concat([df_matches["home"], df_matches["away"]]).unique()

for c in countries:
    if c != country:
        opponents.append(c)

for index, row in df_matches_my_country.iterrows():
    wyscout.append(row["wyscout"])
    skillcorner.append(row["skillcorner"])

df_matches_other_countries = df_matches[(df_matches["home"].isin(opponents)) | (df_matches["away"].isin(opponents))]

for index, row in df_matches_other_countries.iterrows():
    wyscout.append(row["wyscout"])
    skillcorner.append(row["skillcorner"])



# unzip the files in skillcorner

for i in range(len(skillcorner)):
    if np.isnan(skillcorner[i]):
        continue
    skillcorner[i] = str(int(skillcorner[i]))
    try:
        #check if directory exists
        if not os.path.exists(skillcorner_directory + skillcorner[i]):
            with zipfile.ZipFile(skillcorner_directory + str(skillcorner[i]) + ".zip", 'r') as zip_ref:
                zip_ref.extractall(skillcorner_directory + str(skillcorner[i]) + "/")
    except FileNotFoundError:
        print("File not found: " + skillcorner[i])
        skillcorner[i] = np.nan

# remove all zip files
for file in os.listdir(skillcorner_directory):
    if file.endswith(".zip"):
        os.remove(skillcorner_directory + file)




# unzip the files in wyscout
print(wyscout)
wyscout = [str(int(x)) for x in wyscout]

if not (os.path.exists(wyscout_directory + "2024/" + wyscout[0]) or os.path.exists(wyscout_directory + "unl2025-md1+2/" + str(wyscout[0]))):
    with zipfile.ZipFile(wyscout_directory + "2024.zip", 'r') as zip_ref:
        zip_ref.extractall(wyscout_directory + "2024/")

    # needs to be adapted for new data
    with zipfile.ZipFile(wyscout_directory + "unl2025-md1+2.zip", 'r') as zip_ref:
        zip_ref.extractall(wyscout_directory + "2025/")
    with zipfile.ZipFile(wyscout_directory + "unl2025-md3+4.zip", 'r') as zip_ref:
        zip_ref.extractall(wyscout_directory + "2025/")

# remove all the json files not needed in 2024 directory
for file in os.listdir(wyscout_directory + "2024/"):
    if file[:-5] not in wyscout:
        os.remove(wyscout_directory + "2024/" + file)

# remove all the json files not needed in 2025 directory
for file in os.listdir(wyscout_directory + "2025/"):
    if file[:-5] not in wyscout:
        os.remove(wyscout_directory + "2025/" + file)