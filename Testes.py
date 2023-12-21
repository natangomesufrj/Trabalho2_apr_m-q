import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import math

dados = dados = pd.read_csv('conjunto_de_treinamento.csv', delimiter = ',', decimal = '.')

#--------------------------------------------------------
#Tratamento dos dados
#--------------------------------------------------------

#print(dados.iloc[4004])


dados_semid = dados.drop(columns=['Id','tipo','diferenciais','bairro','piscina','churrasqueira','s_jogos','s_festas','s_ginastica','quadra'])
dados_semid = pd.get_dummies(dados_semid,['tipo_vendedor'])


dados = dados.drop([6,405,4004,2568])

#Embaralhando os dados
dados_embaralhados = dados_semid.sample(frac=1,random_state=1)

#Criando os arrays x e y
x_treino = dados_embaralhados.drop(columns='preco')
y_treino = dados_embaralhados.iloc[:3512,:]
x_treino = x_treino.iloc[:3512,:].values
y_treino = y_treino['preco'].values
x_teste = dados_embaralhados.drop(columns='preco')
x_teste = x_teste.iloc[3512:,:].values
y_teste = dados_embaralhados.iloc[3512:,:]
y_teste = y_teste['preco'].values

#Plotando gráficos
x = 10
#plt.scatter(x_treino[:,x],y_treino,color='g')
#plt.plot(x_teste[:,x],y_teste,color='k')
#plt.show()



#--------------------------------------------------------
#Testanto os regressores
#--------------------------------------------------------

print("Linear")

regressor = LinearRegression()
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)

acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

print("KNN")
from sklearn.neighbors import KNeighborsRegressor
for k in range(1,2):
    regressor = KNeighborsRegressor(n_neighbors=k)
    regressor = regressor.fit(x_treino,y_treino)
    y_treino_resultado = regressor.predict(x_treino)
    y_teste_resultado  = regressor.predict(x_teste)

    acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
    acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

    print(k,"%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))


print("SGD")
from sklearn.linear_model import SGDRegressor

regressor = SGDRegressor(loss='squared_epsilon_insensitive', alpha=0.1, penalty='l2')
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)

acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

print("Extra Trees")

from sklearn.ensemble import ExtraTreesRegressor

regressor = ExtraTreesRegressor()
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)

acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

print("Random Forest")
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)

acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

print("Ridge")
from sklearn.linear_model import Ridge
regressor = Ridge()
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)

acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

print("Lasso")
from sklearn.linear_model import Lasso
regressor = Lasso()
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)

acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

print("Gradient Boosting Regressor")
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()
regressor = regressor.fit(x_treino,y_treino)
y_treino_resultado = regressor.predict(x_treino)
y_teste_resultado  = regressor.predict(x_teste)


acuracia_treino = math.sqrt(mean_squared_error(y_treino_resultado,y_treino))
acuracia_teste  = math.sqrt(mean_squared_error(y_teste_resultado,y_teste))

print("%3.1f" % (acuracia_treino),"%3.1f" % (acuracia_teste))

