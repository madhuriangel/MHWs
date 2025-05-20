###This codesame thing doing filtering the longest and strongest from all the events we got
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV again (assuming this is the full event list for both strongest and longest)
df_all = pd.read_csv("mhw_results/all_mhw_events_by_grid.csv")

# Extract year from start date
df_all['Start Date'] = pd.to_datetime(df_all['Start Date'])
df_all['Year'] = df_all['Start Date'].dt.year

# Define "strongest" as top 1 event per grid point by max intensity
df_strongest = df_all.sort_values(['Latitude', 'Longitude', 'Max Intensity (°C)'], ascending=False).drop_duplicates(subset=['Latitude', 'Longitude'])
# Define "longest" as top 1 event per grid point by duration
df_longest = df_all.sort_values(['Latitude', 'Longitude', 'Duration (days)'], ascending=False).drop_duplicates(subset=['Latitude', 'Longitude'])

# Count events per year
strongest_per_year = df_strongest['Year'].value_counts().sort_index()
longest_per_year = df_longest['Year'].value_counts().sort_index()

# Plot
plt.figure(figsize=(14, 6))
plt.bar(strongest_per_year.index - 0.2, strongest_per_year.values, width=0.4, label='Strongest Events', color='crimson')
plt.bar(longest_per_year.index + 0.2, longest_per_year.values, width=0.4, label='Longest Events', color='darkgreen')
plt.xlabel("Year")
plt.ylabel("Number of Events")
plt.title("Yearly Count of Strongest and Longest MHW Events (Alter Method)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
