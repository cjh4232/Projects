import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load rankings data here:
wood_df = pd.read_csv('Golden_Ticket_Award_Winners_Wood.csv')
steel_df = pd.read_csv('Golden_Ticket_Award_Winners_Steel.csv')

wood_steel_df = pd.concat([wood_df,steel_df])

# write function to plot rankings over time for 1 roller coaster here:
def rank_plot (name, park, df):
  
  dff = df[(df['Name'] == name) & (df['Park'] == park)].reset_index()
  rank = dff['Rank'].values
  years = dff['Year of Rank'].values
    
  plt.clf()
  plt.plot(years, rank, marker='o')
  plt.title('{} Ranking Over Time'.format(name))
  plt.xlabel('Years')
  plt.ylabel('Rank')
  plt.show()
    

#rank_plot ('El Toro', 'Six Flags Great Adventure', wood_steel_df)

plt.clf()

# write function to plot rankings over time for 2 roller coasters here:
def rank_plot2 (names, parks, df):
  
  rank1 = df[(df['Name'] == names[0]) & (df['Park'] == parks[0])].Rank.values
  rank2 = df[(df['Name'] == names[1]) & (df['Park'] == parks[1])].Rank.values
  years1 = df[(df['Name'] == names[0]) & (df['Park'] == parks[0])]['Year of Rank'].values
  years2 = df[(df['Name'] == names[1]) & (df['Park'] == parks[1])]['Year of Rank'].values
  
  plt.clf()
  plt.plot(years1, rank1, marker='o', label=names[0], color='green')
  plt.plot(years2, rank2, marker='o', label=names[1], color='blue')
  plt.title('{} and {} Ranking Over Time'.format(names[0], names[1]))
  plt.xlabel('Years')
  plt.ylabel('Rank')
  plt.show()

#rank_plot2 (['El Toro', 'Boulder Dash'], ['Six Flags Great Adventure', 'Lake Compounce'], wood_steel_df)

plt.clf()

# write function to plot top n rankings over time here:
def top_n_ranks(df, n):

  top_n_rankings = df[df['Rank'] <= n]
  
  ax=plt.subplot()

  for coaster in set(top_n_rankings['Name']):
    coaster_rankings = top_n_rankings[top_n_rankings['Name'] == coaster]
    ax.plot(coaster_rankings['Year of Rank'],coaster_rankings['Rank'],label=coaster)
  
  plt.title('Top {} Rollercoasters'.format(n))
  plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
  
  plt.ylabel('Rank')
  plt.xlabel('Year')
  
  plt.show()

#top_n_ranks(wood_steel_df, 3)

plt.clf()

# load roller coaster data here:

rc_df = pd.read_csv('roller_coasters.csv')
print(rc_df.head())
print(rc_df.columns)
# print(rc_df.describe())

# write function to plot histogram of column values here:

def find_num_inversion(df, park_name):

  park_coaster = df[df['park'] == park_name]
  x_values = range(len(inver))

  ax = plt.subplot()
  
  plt.bar(x_values, inver)
  ax.set_xticks(x_values)
  plt.show()


find_num_inversion(rc_df)






plt.clf()

# write function to plot inversions by coaster at a park here:










plt.clf()

# write function to plot pie chart of operating status here:










plt.clf()

# write function to create scatter plot of any two numeric columns here:










plt.clf()
