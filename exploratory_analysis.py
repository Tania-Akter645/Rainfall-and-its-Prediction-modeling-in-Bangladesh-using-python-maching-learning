import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="darkgrid")
os.makedirs('report/figures', exist_ok=True)
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    df = df[['Station Names', 'YEAR', 'Month', 'Max Temp', 'Min Temp',
             'Relative Humidity', 'Rainfall']].copy()
    df.columns = ['station', 'year', 'month', 'max_temp', 'min_temp', 'humidity', 'rainfall']
    df.dropna(inplace=True)
    df = df.sort_values(by=['station', 'year', 'month'])
    df.reset_index(drop=True, inplace=True)
    df = df[(df['year'] >= 2010) & (df['year'] <= 2012)]

    return df

df = load_and_clean_data('data/65 years Weather Data Bangladesh.csv')
#Total Rainfall by Year
colors = ['skyblue', 'orange', 'lightgreen']
rain_by_year = df.groupby('year')['rainfall'].sum()
print("\n🌧️ Total Rainfall by Year:")
print(rain_by_year)

# Plot
rain_by_year.plot(kind='bar', title='Total Rainfall by Year', ylabel='Rainfall (mm)', xlabel='Year', color= colors)
plt.tight_layout()
plt.savefig('report/figures/Total Rainfall by Year.png') # Save figure
plt.show()
#Monthly Average Rainfall
rain_by_month = df.groupby('month')['rainfall'].mean()
print("\n📆 Monthly Average Rainfall:")
print(rain_by_month)

# Plot
rain_by_month.plot(kind='line', marker='o', title='Monthly Average Rainfall (2010–2012)', ylabel='Avg Rainfall (mm)', xlabel='Month')
plt.xticks(range(1,13))
plt.grid(True)
plt.tight_layout()
plt.savefig('report/figures/Monthly Average Rainfall (2010-2012).png') # Save figure
plt.show()
#Top 10 Rainiest Station
colors = ['skyblue', 'orange', 'lightgreen', 'salmon', 'gold',
          'orchid', 'turquoise', 'khaki', 'tomato', 'mediumseagreen']
rain_by_station = df.groupby('station')['rainfall'].sum().sort_values(ascending=False)
print("\n📍 Top 10 Rainiest Stations:")
print(rain_by_station.head(10))

# Plot
rain_by_station.head(10).plot(kind='bar', title='Top 10 Rainiest Stations (2010–2012)', ylabel='Total Rainfall (mm)', color= colors)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/Top 10 Rainiest Stations (2010–2012).png') # Save figure
plt.show()
#Correlation HeatMap
correlation = df[['max_temp', 'min_temp', 'humidity', 'rainfall']].corr()
print("\n🔗 Correlation Matrix:")
print(correlation)

# Heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('report/figures/Correlation Matrix.png')  # Save figure
plt.show()

