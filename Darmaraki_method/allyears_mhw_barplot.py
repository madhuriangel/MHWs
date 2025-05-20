import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV again (assuming this is the full event list for both strongest and longest)
df_all = pd.read_csv("mhw_results/all_mhw_events_by_grid.csv")

# Group all MHW events (not just strongest/longest) by year
df_all['Start Date'] = pd.to_datetime(df_all['Start Date'])
df_all['Year'] = df_all['Start Date'].dt.year

# Count all events per year
all_events_per_year = df_all['Year'].value_counts().sort_index()

# Plot all events per year
# plt.figure(figsize=(14, 6))
# plt.bar(all_events_per_year.index, all_events_per_year.values, color='teal')
# plt.xlabel("Year")
# plt.ylabel("Number of MHW Events")
# plt.title("Yearly Count of All Marine Heatwave Events (Alter Method)")
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# Plot again with all years shown on x-axis
plt.figure(figsize=(14, 6))
plt.bar(all_events_per_year.index, all_events_per_year.values, color="#005E63")
plt.xlabel("Year")
plt.ylabel("Number of MHW Events")
plt.title("Yearly Count of All Marine Heatwave Events (Darmaraki)")
plt.xticks(ticks=all_events_per_year.index, rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
