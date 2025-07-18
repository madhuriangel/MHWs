import pandas as pd
import os

# Load the full events CSV
df = pd.read_csv('mhw_analysis_alterhob/mhw_results/all_mhw_events_by_grid.csv', parse_dates=['Start Date', 'End Date'], dayfirst=True)

# 1) Filter for 2021, July–October
mask_2021 = (
    (df['Start Date'].dt.year == 2021) &
    (df['Start Date'].dt.month >= 7) &
    (df['Start Date'].dt.month <= 10)
)
df_2021 = df.loc[mask_2021].reset_index(drop=True)
out_2021 = 'mhw_analysis_alterhob/mhw_results/events_2021_jul_oct.csv'
df_2021.to_csv(out_2021, index=False)

# 2) Filter for 2023, January and June–September
mask_2023 = (
    (df['Start Date'].dt.year == 2023) &
    (
        (df['Start Date'].dt.month == 1) |
        ((df['Start Date'].dt.month >= 6) & (df['Start Date'].dt.month <= 9))
    )
)
df_2023 = df.loc[mask_2023].reset_index(drop=True)
out_2023 = 'mhw_analysis_alterhob/mhw_results/events_2023_jan_jun_sept.csv'
df_2023.to_csv(out_2023, index=False)

# Confirm files created
print("Saved files:")
print(f" - {out_2021}")
print(f" - {out_2023}")