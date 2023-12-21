import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

dados = pd.read_csv('conjunto_de_treinamento.csv', delimiter = ',', decimal = '.')
final = pd.read_csv('conjunto_de_teste.csv', delimiter = ',', decimal = '.')

dados.to_csv('b.csv')

#--------------------------------------------------------
#Tratamento dos dados
#--------------------------------------------------------

dados_semid = dados.drop(columns=['Id','tipo','diferenciais','bairro','piscina','churrasqueira','s_jogos','s_festas','s_ginastica','quadra'])
dados_semid = pd.get_dummies(dados_semid,['tipo_vendedor'])
final_semid = final.drop(columns=['Id','tipo','diferenciais','bairro','piscina','churrasqueira','s_jogos','s_festas','s_ginastica','quadra'])
final_semid = pd.get_dummies(final_semid,['tipo_vendedor'])



dados_semid = dados_semid.drop([6,405,4004,2568])

dados_embaralhados = dados_semid.sample(frac=1,random_state=54319)

x = dados_embaralhados.drop(columns='preco')
x_treino = x.values
y_treino = dados_embaralhados['preco'].values
x_final = final_semid.values

#--------------------------------------------------------
#O regressor
#--------------------------------------------------------

regressor = LinearRegression()
regressor = regressor.fit(x_treino,y_treino)
y_final_resultados  = regressor.predict(x_final)

#--------------------------------------------------------
#Escrevendo no arquivo
#--------------------------------------------------------]

id = []
for a in range(2000):
    id.append(a)

id = np.array(id)

resultado = pd.DataFrame({'Id': id, 'preco': y_final_resultados})

resultado.to_csv('resultado_2.csv', index=False)
