import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression
#choosing AXIS bank companies with AXIS as main ICICI and AXIS bank as compatetors along with NIFTY 50 as market index 
stocks = [
    "AXISBANK.NS",
    "ICICIBANK.NS",
    "HDFCBANK.NS",
    "KOTAKBANK.NS",
    "^NSEI"
]
# Download data (2 years of daily data)
end_date=datetime.today()
start_date=end_date-timedelta(days=2*365)
close_df=pd.DataFrame()
for stock in stocks:
  data=yf.download(stock,start=start_date,end=end_date)
  close_df[stock]=data['Close']
#using this to clean the data cause NaN values will be there in the data set and pct_change is for daily returns
close_df=close_df.pct_change().dropna()
#now using seaborn ploting regression
plt.figure(figsize=(12,6))
sns.regplot(x=close_df['^NSEI'],y=close_df['AXISBANK.NS'],scatter=True,color='blue')
plt.xlabel('Market Index')
plt.ylabel('AXIS bank')
plt.title('AXIS bank vs Market Index')
plt.show()
#calculating beta and alpha using linear progression
#y=bx+a   here beta is slope and alpha is intecept 
#using linear regression our data would be split in 2 part train and test in 2:1.
#Our goal is to find a st line so that differnce observed data points and the predicted values is minimise the error least square 
model=LinearRegression()
X=close_df['^NSEI'].values.reshape(-1,1)
y=close_df['AXISBANK.NS']
model.fit(X,y)
beta=model.coef_[0]
alpha=model.intercept_
print(f'Beta: {beta}')
print(f'Alpha: {alpha}')
#now atlast correlation matrix.This tells about dependence of 2 stocks
# for eg A,B be 2 stocks let corr=1 this means A and B will follow same behaviour.If corr=-1 the behaivour will be exactly opposite and corr=0 means no relation
correlation_matrix=close_df.corr()
#correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
