import csv
import pandas as pd
import os 
import datetime
import pathlib

df = pd.read_csv('experiments2.csv', index_col=False,sep=",")
df1 = df[df["yes"]!=0]
#df1 = df1.drop("axis = 1)


cols = df1.columns
for i in range(df1.shape[0]):
    command = ""
    stamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    output_name = ""
    for c in cols:
        if c != "file" and c != "yes" :
            # if c in ["WANTED_TESTS", "WANTED_FEATURES", "WANTED_ALGS"]:
            #     command += f" --{c} '{df1.iloc[i][c]}'"
            # el
            if c == "output":
                output_name = f"{df1.iloc[i]['output']}_{stamp}_rowId{i}"
                command += f" --{c} {output_name}"
            else:
                command += f" --{c} {df1.iloc[i][c]}"
                
    command = df1.iloc[i]["file"] + command
    
    current_model = f"logs"
    pathlib.Path(f'{current_model}/').mkdir(parents=True, exist_ok=True)#metrics

    command = f"nohup python3 -u {command} > {current_model}/{output_name}.out"
    print(command)
    os.system(command)
