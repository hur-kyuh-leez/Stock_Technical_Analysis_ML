# Purpose
Technical_Analysis is used to build the model presented in 'A study on KOSPI 200 direction forecasting using XGBoost model' by Dae Woo Hah1 · Young Min Kim2 · Jae Joon Ahn3
#### Note  
* Please note that only XGBoost was used here since the error happened during data pre-processing.
* Please note basic ML parameters were used since using proper features was more important than choosing optimized parameters.
* Please note the data was downloaded from Investing.com for research purposes only.
* Please note that longer time series is used to magnify errors.

## Findings
Due to inappropriate data pre-processing, the model uses 20 days of future data. That's why 20 days window or shifting 1~20 days worked the best.
#### Possible mistakes in details
The authors dropped NaN too early. Then, they attached pd.series to pd.Dataframe without appropriate merging.
## Paper 
* [A study on KOSPI 200 direction forecasting using XGBoost model by Dae Woo Hah1 · Young Min Kim2 · Jae Joon Ahn3](http://www.kdiss.org/journal/view.html?uid=2503&&vmd=Full)



## Contributor
Hyuk Kyu Lee
* [Technical_Analysis_ML](Technical_Analysis_ML.py)
