import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Station Names', 'YEAR', 'Month', 'Max Temp', 'Min Temp',
             'Relative Humidity', 'Rainfall']].copy()
    df.columns = ['station', 'year', 'month', 'max_temp', 'min_temp', 'humidity', 'rainfall']
    df.dropna(inplace=True)
    df = df[(df['year'] >= 2010) & (df['year'] <= 2012)]
    return df


df = load_and_clean_data('data/65 years Weather Data Bangladesh.csv')


station_rain = df.groupby('station')['rainfall'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 8))
plt.pie(station_rain, labels=station_rain.index, autopct='%1.1f%%',
        startangle=140, colors=plt.cm.tab10.colors)
plt.title('Top 10 Rainiest Stations (2010–2012)')
plt.axis('equal')
plt.tight_layout()

plt.savefig('report/figures/rainiest_station_pie_chart.png') #save figure
plt.show()

