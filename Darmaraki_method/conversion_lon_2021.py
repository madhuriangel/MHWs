import pandas as pd

# 1. Load your toxicity‐site file
# Replace 'your_tox.csv' with the actual path
df = pd.read_csv('mhw_analysis_alterhob/hab2021.csv', parse_dates=['periodstart_date','periodend_date'], dayfirst=True)

# 2. Round Latitude
df['Latitude'] = df['Latitude'].round(3)

# 3. Convert Longitude to 0–360 and round
df['Longitude'] = df['Longitude'].apply(lambda x: x + 360 if x < 0 else x).round(3)

# 4. Save
df.to_csv('mhw_analysis_alterhob/hab2021_conv.csv', index=False)

# 5. (Optional) Inspect
print(df.head())
