import sec_10q_data as s10q
import sec_10q_etl as etl10q
from sec_10q_etl import fill_lr
import yfinance as yf
import pandas as pd
import datetime

tickers_energies =  [
    'XOM',   # Exxon Mobil Corporation
    'CVX',   # Chevron Corporation
    'BP',    # BP plc
    'TTE',   # TotalEnergies SE
    'SHEL',  # Shell plc
    'COP',   # ConocoPhillips
    'SLB',   # Schlumberger Limited
    'HAL',   # Halliburton Company
    'PSX',   # Phillips 66
    'MPC',   # Marathon Petroleum Corporation
    'KMI',   # Kinder Morgan, Inc.
    'EOG',   # EOG Resources, Inc.
    'PXD',   # Pioneer Natural Resources Company
    'OXY',   # Occidental Petroleum Corporation
    'DVN',   # Devon Energy Corporation
    'BKR',   # Baker Hughes Company
    'VLO',   # Valero Energy Corporation
    'LNG',   # Cheniere Energy, Inc.
    'EQNR',  # Equinor ASA
    'ENB',   # Enbridge Inc.
]

dict_banks = {}

#if you don't have the .csv files, run code between lines 33 and 40. It take some minutes.
# for tick in tickers_energies:
#     try:
#         dict_banks[tick] = etl10q.fill_lr(etl10q.ten_q_2_df(s10q.get_10q_data(tick, 'eng.gabrielcoutinho@outlook.com.br'), minimum_records=12))
#         print(dict_banks[tick].head())
#         dict_banks[tick].to_csv(tick + '.csv', sep=';')
#     except:
#         continue


for tick in tickers_energies:
    try:
        # print(yf.Ticker(tick).history(start='2024-09-28', end='2024-10-05')['Close'][0])
        df = pd.read_csv(tick + '.csv', sep=';')#.set_index('end')
        #print(df.head())
        df['end'] = df['end'].apply(lambda x: datetime.date(year=int(x.split('-')[0]), month=int(x.split('-')[1]), day=int(x.split('-')[2])))
        df['StockPrice'] = df.apply(lambda x: yf.Ticker(tick).history(start=x['end'], end=(x['end'] + datetime.timedelta(days=10)))['Close'].iloc[0], axis=1)
        df.set_index('end', inplace=True)
        df.to_csv(tick + '_with_price.csv', sep=';')
    except:
        print('Couldnt do me: ', tick)
