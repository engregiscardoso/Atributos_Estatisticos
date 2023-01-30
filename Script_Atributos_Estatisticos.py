# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:35:58 2023

@author: REGIS CARDOSO
"""


######################################################################################################
## ALGORITMO PARA FEATURE ENGINEERING UTILIZANDO ATRIBUTOS ESTATÍSTICOS E FFT ###
######################################################################################################

## IMPORTAR AS BIBLIOTECAS UTILIZADAS ###

import pandas as pd
import numpy as np
import statistics
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from scipy.fftpack import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


## IMPORTAR OS ARQUIVOS DE DADOS ###

df = pd.read_csv('Dado_Vibracao.csv', sep=';')

df.columns = ['Tempo', 'Valor']

columns = ['Tempo', 'Valor']



## CONVERTE OS DADOS EM FLOAT E ADICIONA O PONTO COMO SEPARADOR DECIMAL ###

df[columns] = df[columns].apply(lambda x: x.str.replace(',', '.').astype('float'))


## FUNÇÕES ###

# FUNÇÃO PARA FEATURE ENGINEERING VIA ATRIBUTOS ESTATÍSTICOS

def atributos_estatisticos(df):
    max_n = df['Valor'].max()
    min_n = df['Valor'].min()
    skew = df['Valor'].skew(axis=0, skipna=True)
    mean = df['Valor'].mean(axis=0, skipna=True)
    median = df['Valor'].median(axis=0, skipna=True)
    std = df['Valor'].std(axis=0, skipna=True)
    var = df['Valor'].var(axis=0, skipna=True)
    kurt = df['Valor'].kurt(axis=0, skipna=True)

    #moda = statistics.stats(df['Valor'])
    
    return max_n, min_n, skew, mean, median, std, var, kurt



## IMPORTAR OS ARQUIVOS DE DADOS ###

df = pd.read_csv('Dado_Vibracao.csv', sep=';')

df.columns = ['Tempo', 'Valor']

columns = ['Tempo', 'Valor']


## CONVERTE OS DADOS EM FLOAT E ADICIONA O PONTO COMO SEPARADOR DECIMAL ###

df[columns] = df[columns].apply(lambda x: x.str.replace(',', '.').astype('float'))


## APLICANDO A FUNÇÃO DE FEATURE ENGINEERING VIA ATRIBUTOS ESTATÍSTICOS  ###
    
max_n, min_n, skew, mean, median, std, var, kurt = atributos_estatisticos(df)

columns_estat = ['max', 'min', 'skew', 'mean','median', 'std', 'var', 'kurt']

df_atributos_estatisticos = []

df_atributos_estatisticos = pd.DataFrame(df_atributos_estatisticos, columns = columns_estat)

df_atributos_estatisticos = df_atributos_estatisticos.append({'max':  max_n, 'min': min_n, 'skew': skew, 'mean': mean, 'median': median, 'std': std, 'var': var, 'kurt': kurt} , ignore_index=True)
