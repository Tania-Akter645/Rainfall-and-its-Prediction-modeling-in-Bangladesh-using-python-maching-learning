import pandas as pd

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

if __name__ == "__main__":
    df = load_and_clean_data('data/65 years Weather Data Bangladesh.csv')


    print(df)



