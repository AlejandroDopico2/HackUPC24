import numpy
import pandas as pd


def clean_dataset(csv):
    """"
    Reformat the dataset adding columns year, product type, season and section
    :param csv: dataset
    :return: dataset with new columns
    """

    # example of image: https://static.zara.net/photos///2024/S/0/2/...

    # open the dataset
    df = pd.read_csv(csv)

    for i in range(len(df)):
        sample = df.iloc[i,0]
        print(sample)

        if isinstance(sample, str):
            year, season, product_type, section = sample.split('///')[1].split('/')[:4]

            # add the new columns
            df.loc[i, 'year'] = year
            df.loc[i, 'season'] = season
            df.loc[i, 'product_type'] = product_type

    df.to_csv('inditex_tech_data_formatted.csv', index=False)




if __name__ == '__main__':
    clean_dataset('inditex_tech_data.csv')