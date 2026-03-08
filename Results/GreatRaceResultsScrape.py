import pandas as pd
import glob
import os

# Folder containing your STAGE files
data_dir = 'Results'

# Grab all files that start with STAGE
files = glob.glob(os.path.join(data_dir, "STAGE*.htm"))

all_dfs = []

for file in files:
    # Extract stage number from filename
    stage_name = os.path.basename(file).replace(".htm", "")
    
    # Read all tables in the HTML
    tables = pd.read_html(file)
    
    # Your result table is the first table in these files
    df = tables[0]
    
    # Add stage identifier column
    df["Stage"] = stage_name
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    all_dfs.append(df)

# Combine all stages
combined_df = pd.concat(all_dfs, ignore_index=True)

# Optional: clean time fields (remove leading *)
# time_cols = [c for c in combined_df.columns if "LEG" in c or c in ["TOTAL", "SCORE"]]
# for col in time_cols:
#     combined_df[col] = combined_df[col].astype(str).str.replace("*", "", regex=False)

# Drop unwanted columns
combined_df = combined_df.drop(columns=[c for c in combined_df.columns if c in ["Unnamed: 15", "Unnamed: 16"]])

# Rename first column
combined_df = combined_df.rename(columns={"RANK+P1KA1:S1": "RANK"})

# Reorder columns
cols_order = [
    "RANK", "CAR", "YEAR", "ScYR", "DIV", "CREW",
    "LEG 1", "LEG 2", "LEG 3", "LEG 4", "LEG 5", "LEG 6",
    "PENALTY", "TOTAL", "FACTOR", "SCORE",
    "Stage"
]

combined_df['Stage'] = combined_df['Stage'].str.extract(r'(\d+)').astype(int)

# Keep only the columns that exist in the DataFrame
cols_order = [c for c in cols_order if c in combined_df.columns]

combined_df = combined_df[cols_order]

# Export to Parquet
combined_df.to_parquet(".\\Output\\great_race_all_stages.parquet", index=False)

# Export to Excel
combined_df.to_excel(".\\Output\\great_race_all_stages.xlsx", index=False)

print("Done.")