import pandas as pd, numpy as np
pd.set_option('display.expand_frame_repr', False)
import xgboost as xg
from sklearn.model_selection import train_test_split  # training and testing data split
from sklearn.model_selection import cross_val_score  # score evaluation
from sklearn import metrics  # accuracy measure
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pickle
import sys


"""
Author: Hyuk Kyu Lee
Purpose: Technical_Analysis.py is used to build the model presented in 'A study on KOSPI 200 direction forecasting using XGBoost model' by Dae Woo Hah1 · Young Min Kim2 · Jae Joon Ahn3
Source: http://www.kdiss.org/journal/view.html?uid=2503&&vmd=Full
Findings: Due to inappropriate data pre-processing, the model used 20 days of future data. That's why 20 days window or shifting 1~20 days worked the best.
Possible mistakes in details: The authors dropped NaN too early. Then, they attached pd.series to pd.Dataframe without appropriate merging method.
Note:
Please note that only XGBoost was used here since the error happened during data pre-processing.
Please note basic ML parameters were used since using proper features was more important than choosing optimized parameters.
Please note the data was downloaded from Investing.com for research purposes only.
Please note that longer time series is used to magnify errors.
"""

class Technical_Analysis:
    def __init__(self, filename, save_Bool=True, predicting_Bool=False, test_size=0.20):
        self.filename = filename
        self.save_Bool = save_Bool
        self.test_size = test_size
        self.predicting_Bool = predicting_Bool

    # credit to: https://wikidocs.net/3397
    def add_fnMACD(self, m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
        m_Df['EMAFast'] = m_Df['Price'].ewm(span=m_NumFast, min_periods=m_NumFast - 1).mean()
        m_Df['EMASlow'] = m_Df['Price'].ewm(span=m_NumSlow, min_periods=m_NumSlow - 1).mean()
        m_Df['MACD'] = m_Df['EMAFast'] - m_Df['EMASlow']
        m_Df['MACD'] = m_Df['MACD'].ewm(span=m_NumSignal, min_periods=m_NumSignal-1).mean()
        return m_Df

    # credit to: https://wikidocs.net/3399
    def add_fnRSI(self, m_Df, m_N=20):
        U = np.where(m_Df['Price'].diff(1) > 0, m_Df['Price'].diff(1), 0)
        D = np.where(m_Df['Price'].diff(1) < 0, m_Df['Price'].diff(1) * (-1), 0)
        AU = pd.DataFrame(U).rolling(window=m_N, min_periods=m_N).mean()
        AD = pd.DataFrame(D).rolling(window=m_N, min_periods=m_N).mean()
        RSI = AU.div(AD+AU) * 100
        m_Df['RSI'] = RSI
        return RSI

    # credit to https://wikidocs.net/3396
    def add_fnStoch(self, m_Df, n=20): # price: 종가(시간 오름차순), n: 기간
        sz = len(m_Df['Price'])
        if sz < n:
            # show error message
            raise SystemExit('입력값이 기간보다 작음')
        tempSto_K=[]
        for i in range(sz):
            if i >= n-1:
                tempUp = m_Df['Price'][i] - min(m_Df['Low'][i-n+1:i+1])
                tempDown = max(m_Df['High'][i-n+1:i+1]) - min(m_Df['Low'][i-n+1:i+1])
                tempSto_K.append(tempUp / tempDown)
            else:
                tempSto_K.append(0) #n보다 작은 초기값은 0 설정
        m_Df['Sto_K'] = pd.Series(tempSto_K,  index=m_Df.index)
        m_Df['Sto_D'] = m_Df['Sto_K'].rolling(3).mean()
        return m_Df

    def shifting(self, days):
        self.df[[f'SMA(20)_shift{days}', f'EMA(20)_shift{days}', f'Disparity_shift{days}', f'MACD_shift{days}', f'RSI_shift{days}', f'Sto_K_shift{days}',
            f'Sto_D_shift{days}']] = self.df[['SMA(20)', 'EMA(20)', 'Disparity', 'MACD', 'RSI', 'Sto_K', 'Sto_D']].shift(days)

    def find_sharpe_ratio(self, series):
        expected_return = pd.DataFrame(np.cumsum(series)).diff()
        risk_free_return = 0
        self.sharpe_ratio = float((expected_return.mean() - risk_free_return) / expected_return.std() * np.sqrt(252))

    def predicting(self, df_X, saved_model_path):
        d = {}
        df_X = X.tail(1)
        for index, column_name in enumerate(df_X.columns):
            d[column_name] = [df_X[column_name].values[0]]

        test_X = pd.DataFrame(data=d)
        filename = saved_model_path.split('/')[-1]
        loaded_model = pickle.load(open(f'{saved_model_path}', 'rb'))
        predictions = loaded_model.predict(test_X)
        print(d)
        print(f'\n{filename} predictions...{predictions[0]}')
        sys.exit()

    def save_df(self, df):
        df.to_csv(f"data/byproduct_data/{self.filename}_data_prep_checking.csv", encoding='utf-8',
                  index=False)
        sys.exit()

    def csv_to_df(self):
        print(f'Technical Analysis on...{self.filename}')
        df = pd.read_csv(f"data/{self.filename}.csv",
                         thousands=',', parse_dates=[0])
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        self.df = df

    def add_SMA(self):
        self.df['SMA(20)'] = self.df.Price.rolling(20).mean()

    def add_EMA(self):
        self.df['EMA(20)'] = self.df['Price'].ewm(span=20, min_periods=20, adjust=True).mean()

    def add_Disparity(self):
        self.df['Disparity'] = self.df['Price'] / self.df['SMA(20)'] * 100

    def add_Close_to_Close(self):
        self.df['Close_Close'] = self.df['Price'].pct_change()
        self.df.loc[(self.df['Close_Close'] < 0), 'Close_Close_cat'] = 0
        self.df.loc[(self.df['Close_Close'] > 0), 'Close_Close_cat'] = 1
        self.df.loc[(self.df['Close_Close'] == 0), 'Close_Close_cat'] = 2
        self.df['Close_Close'] = self.df['Close_Close'].shift(-1) # Shifting to predict the future, also note that df is in ascending order
        self.df['Close_Close_cat'] = self.df['Close_Close_cat'].shift(-1) # Shifting to predict the future, also note that df is in ascending order

    # basic_testing() assumes agent(investor) fully uses its capital each time.
    def basic_testing(self, df, predictions, buy_only, sell_only, graphing):
        df['Prediction'] = predictions
        df = df[['Date', df.columns[1], df.columns[2], 'Prediction']]
        if buy_only:
            df.loc[df['Prediction'] == 1, 'Virtual_Trade'] = df[df.columns[1]]
        if sell_only:
            df.loc[df['Prediction'] == 0, 'Virtual_Trade'] = -df[df.columns[1]]

        df.loc[df[df.columns[2]] == df[df.columns[3]], f'Test_Result'] = 1
        df.loc[df[df.columns[2]] != df[df.columns[3]], f'Test_Result'] = 0
        df.loc[(df[df.columns[2]] != df[df.columns[3]]) & (df[df.columns[3]] == 1), f'Wrong_Sum'] = df[
            df.columns[1]]
        df.loc[(df[df.columns[2]] != df[df.columns[3]]) & (df[df.columns[3]] == 0), f'Wrong_Sum'] = -df[
            df.columns[1]]
        df.loc[(df[df.columns[2]] == df[df.columns[3]]) & (df[df.columns[3]] == 1), f'Right_Sum'] = df[
            df.columns[1]]
        df.loc[(df[df.columns[2]] == df[df.columns[3]]) & (df[df.columns[3]] == 0), f'Right_Sum'] = -df[
            df.columns[1]]
        df = df.fillna(0)
        virtual_trade = df['Virtual_Trade'].values
        self.find_sharpe_ratio(virtual_trade)
        # Plotting
        if graphing:
            plt.plot(np.cumsum(virtual_trade))
            plt.title('Virtual_Trade')
            plt.show()
        return df


if __name__ == "__main__":
    TA_obj = Technical_Analysis(filename='KOSPI')
    TA_obj.csv_to_df()
    TA_obj.add_SMA()
    TA_obj.add_EMA()
    TA_obj.add_Disparity()
    TA_obj.add_Close_to_Close()
    '''
        This is where the Error usually happens: dropping a part of Df and using Pd.series to merge.
        # TA_obj.df = TA_obj.df.dropna()
    '''
    TA_obj.add_fnMACD(TA_obj.df)
    TA_obj.add_fnRSI(TA_obj.df)
    TA_obj.add_fnStoch(TA_obj.df)
    TA_obj.df = TA_obj.df[['Date', 'Close_Close', 'Close_Close_cat', 'SMA(20)', 'EMA(20)', 'Disparity', 'MACD', 'RSI', 'Sto_K', 'Sto_D']] # Dropping unnecessary columns
    TA_obj.df = TA_obj.df.dropna() # Right place to drop NaN

    # Checking DF
    print(TA_obj.df.head(100))
    print(TA_obj.df.tail(100))

    # 20 days window
    for i in range(20):
        TA_obj.shifting(i+1)


    # Machine Learning part, credit to Kaggle.com
    train, test = train_test_split(TA_obj.df, test_size=TA_obj.test_size, shuffle=False)
    train_X = train[train.columns[3:]]
    train_Y = train[train.columns[2:3]]
    test_X = test[test.columns[3:]]
    test_Y = test[test.columns[2:3]]
    X = TA_obj.df[TA_obj.df.columns[3:]]
    Y = TA_obj.df[TA_obj.df.columns[2:3]]

    # Predicting pre-processed data
    if TA_obj.predicting_Bool:
        TA_obj.predicting(X, f'data/byproduct_data/Technical_Analysis_ML_{TA_obj.filename}.sav')


    print('\n\n\nBuilding Model...')
    model = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
    # Cross validation uses many Folds and avg them out, so you can even use this after model.fit because it won't used the fitted model
    result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
    xgb_classifier = model.fit(train_X, train_Y)

    # save the model to disk
    if TA_obj.save_Bool:
        pickle.dump(xgb_classifier, open(f'data/byproduct_data/Technical_Analysis_ML_{TA_obj.filename}.sav', 'wb'))
    predictions = xgb_classifier.predict(test_X)
    xgb_predictions = predictions
    pred_df = TA_obj.basic_testing(df=test, predictions=xgb_predictions, buy_only=True, sell_only=True, graphing=True)
    pred_df.to_csv(f"data/byproduct_data/{TA_obj.filename}_prediction.csv", encoding='utf-8',
              index=False)
    print(f'For {len(predictions)}days')
    print('The accuracy of the XGBoost is', metrics.accuracy_score(predictions, test_Y))
    print('The cross validated score for XGBoost is:', result.mean())



