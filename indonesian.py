import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Reading rows containing cities population
df = pd.read_excel(
    'population.xls',
    header = 3,
    skipfooter = 3,
    index_col = 0,
    na_values = '-'
)

# Reading all cities population including INDONESIA row as an accumulation
df2 = pd.read_excel(
    'population.xls',
    header = 3,
    skipfooter = 2,
    index_col = 0,
    na_values = '-'
)

# City where citizens is the most in 2010:
dfMax2010 = df[df[2010] == df[2010].max()]
namaMax2010 = dfMax2010.index.values[0]    

# City where citizens is the least in 1971:
df = df.dropna(subset = [1971])     # dropping cities where value is NaN in 1971
dfMin1971 = df[df[1971] == df[1971].min()] 
namaMin1971 = dfMin1971.index.values[0]     

# Reading INDONESIA data
dfindo = df2[df2[2010] == df2[2010].max()] 
namaindo = dfindo.index.values[0]     


# Linear Regression
from sklearn.linear_model import LinearRegression
modelMax2010 = LinearRegression()
modelMin1971 = LinearRegression()
modelindo = LinearRegression()

# Training
# ---- Jawa Barat ----  
x = dfMax2010.columns.values.reshape(-1, 1)
y = dfMax2010.values[0]
modelMax2010.fit(x, y)
# ---- Bengkulu ----
x = dfMin1971.columns.values.reshape(-1, 1)
y = dfMin1971.values[0]
modelMin1971.fit(x, y)
# ---- INDONESIA ----
x = dfindo.columns.values.reshape(-1, 1)
y = dfindo.values[0]
modelindo.fit(x, y)

# Prediction of population by 2050
max2050 = int(round(modelMax2010.predict([[ 2050 ]])[0]))
min2050 = int(round(modelMin1971.predict([[ 2050 ]])[0]))
indo2050 = int(round(modelindo.predict([[ 2050 ]])[0]))
print('Prediction of', namaMax2010, 'population in 2050:', max2050)
print('Prediction of', namaMin1971, 'population in 2050:', min2050)
print('Prediction of', namaindo, 'population in 2050:', indo2050)

# Plotting
plt.plot(
    dfMax2010.columns.values, dfMax2010.iloc[0], 'g-',
)
plt.plot(
    dfMin1971.columns.values, dfMin1971.iloc[0], 'm-',
)
plt.plot(
    dfindo.columns.values, dfindo.iloc[0], 'r-',
)

plt.scatter(
    dfMax2010.columns.values, dfMax2010.iloc[0], color = 'g', s = 80,
)       
plt.scatter(
    dfMin1971.columns.values, dfMin1971.iloc[0], color = 'm', s = 80,
)
plt.scatter(
    dfindo.columns.values, dfindo.iloc[0], color = 'r', s = 80,
)

# Plotting Best Fit Line
plt.plot(
    dfMax2010.columns.values,
    modelMax2010.coef_ * dfMax2010.columns.values + modelMax2010.intercept_,
    'y-'
)
plt.plot(
    dfMin1971.columns.values,
    modelMin1971.coef_ * dfMin1971.columns.values + modelMin1971.intercept_,
    'y-'
)
plt.plot(
    dfindo.columns.values,
    modelindo.coef_ * dfindo.columns.values + modelindo.intercept_,
    'y-'
)

plt.legend([namaMax2010, namaMin1971, namaindo])
plt.title('{} Population (1971-2010)'.format(namaindo))
plt.xlabel('Year')
plt.ylabel('Population (hundred million people')
plt.grid(True)

plt.show()


