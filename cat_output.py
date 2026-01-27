import pandas as pd
import os

file_path = "final_pipeline_output.csv"
if os.path.exists(file_path):
    print(f"File {file_path} exists.")
    try:
        df = pd.read_csv(file_path)
        print(df.to_string())
    except Exception as e:
        print(f"Error reading csv: {e}")
else:
    print(f"File {file_path} does not exist.")
