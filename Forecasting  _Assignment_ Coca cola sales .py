#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf


# ## __2 - Data collection and description__ 

# In[2]:


df = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")


# In[3]:


df1 = df.copy()


# In[4]:


df1.head()


# In[5]:


df1.isnull().sum()


# In[6]:


df1.dtypes


# In[7]:


df1.describe().T


# In[8]:


temp = df1.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')


# In[9]:


df1['quater'] = pd.to_datetime(temp).dt.strftime('%b-%Y')


# In[10]:


df1.head()


# In[11]:


df1 = df1.drop(['Quarter'], axis=1)


# In[12]:


df1.reset_index(inplace=True)


# In[13]:


df1['quater'] = pd.to_datetime(df1['quater'])


# In[14]:


df1 = df1.set_index('quater')


# In[15]:


df1.head()


# In[16]:


df1['Sales'].plot(figsize=(15, 6))
plt.show()


# In[17]:


for i in range(2,10,2):
    df1["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[18]:


ts_add = seasonal_decompose(df1.Sales,model="additive")
fig = ts_add.plot()
plt.show()


# In[19]:


ts_mul = seasonal_decompose(df1.Sales,model="multiplicative")
fig = ts_mul.plot()
plt.show()


# In[20]:


tsa_plots.plot_acf(df1.Sales)


# ## __3 - Building Time series forecasting with ARIMA__ 

# In[21]:


X = df1['Sales'].values


# In[22]:


size = int(len(X) * 0.66)


# In[23]:


train, test = X[0:size], X[size:len(X)]


# In[24]:


model = ARIMA(train, order=(5,1,0))


# In[25]:


model_fit = model.fit(disp=0)


# In[26]:


print(model_fit.summary())


# ### This summarizes the coefficient values used as well as the skill of the fit on the on the in-sample observations

# In[27]:


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# ### The plot of the residual errors suggests that there may still be some trend information not captured by the model  
# ### The results show that indeed there is a bias in the prediction (a non-zero mean in the residuals)  

# ### __3.1 - Rolling Forecast ARIMA Model__ 

# In[28]:


history = [x for x in train]


# In[29]:


predictions = list()


# In[30]:


for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# In[31]:


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[32]:


pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# ### A line plot is created showing the expected values (blue) compared to the rolling forecast predictions (red). We can see the values show some trend and are in the correct scale  

# ## __4 - Comparing Multiple Models__ 

# In[33]:


df2 = pd.get_dummies(df, columns = ['Quarter'])


# In[34]:


df2.columns = ['Sales','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4']


# In[35]:


df2.head()


# In[36]:


t= np.arange(1,43)


# In[37]:


df2['t'] = t


# In[38]:


df2['t_sq'] = df2['t']*df2['t']


# In[39]:


log_Sales=np.log(df2['Sales'])


# In[40]:


df2['log_Sales']=log_Sales


# In[41]:


df2.head()


# In[42]:


train1, test1 = np.split(df2, [int(.67 *len(df2))])


# In[43]:


linear= smf.ols('Sales ~ t',data=train1).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test1['t'])))
rmselin=np.sqrt((np.mean(np.array(test1['Sales'])-np.array(predlin))**2))
rmselin


# In[44]:


quad=smf.ols('Sales~t+t_sq',data=train1).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test1[['t','t_sq']])))
rmsequad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predquad))**2))
rmsequad


# In[45]:


expo=smf.ols('log_Sales~t',data=train1).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test1['t'])))
rmseexpo=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo


# In[46]:


additive= smf.ols('Sales~ Q1+Q2+Q3+Q4',data=train1).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test1[['Q1','Q2','Q3','Q4']])))
rmseadd=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predadd))**2))
rmseadd


# In[47]:


addlinear= smf.ols('Sales~t+Q1+Q2+Q3+Q4',data=train1).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test1[['t','Q1','Q2','Q3','Q4']])))
rmseaddlinear=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear


# In[48]:


addquad=smf.ols('Sales~t+t_sq+Q1+Q2+Q3+Q4',data=train1).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test1[['t','t_sq','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predaddquad))**2))
rmseaddquad


# In[49]:


mulsea=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train1).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test1[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul


# In[50]:


mullin= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train1).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test1[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin


# In[51]:


mul_quad= smf.ols('log_Sales~t+t_sq+Q1+Q2+Q3+Q4',data=train1).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test1[['t','t_sq','Q1','Q2','Q3','Q4']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad


# ## __5 - Conclusion__ 

# In[52]:


output = {'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),
          'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}


# In[53]:


rmse=pd.DataFrame(output)


# In[55]:


print(rmse)


# ### Additive seasonality with quadratic trend has the best RMSE value

# In[ ]:




