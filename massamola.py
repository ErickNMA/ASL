# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:45:35 2022

@author: valter

Exemplo de simulação massa-mola
"""

# importa bibliotecas

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp


plt.close('all') # Fecha gráficos

"""
Define EDO - Caso Massa mola

""" 

def dydt(t,F,K,M,B,u): # argumentos A, beta e Fi precisam ser informados!
    y1, y2 = F
    return [y2,
            -(K/M)*y1-(B/M)*y2+u/M]    

# Solução para :

# tempo    
t0 = 0    # tempo inicial
tf = 2.5   # tempo final
t = np.linspace(t0,tf,100) # instantes que desejo ter a solulcao

# parametros do sistema
K = 25
M = 1
B = 5

# condição inicial e valor de Fi (=u)
Y0 = (1, 0.2) # Posição e velocidades iniciais
u = 5     # Força Fi aplicada (sinal de controle u!)

sol1 = solve_ivp(dydt, t_span=(t0,tf), y0=Y0, method='RK23',
                 t_eval=t, args=(K,M,B,u))

y1 = sol1.y[0]
y2 = sol1.y[1]

plt.figure(1)
plt.plot(t, y1, 'b',label='Posição, y_1(t)')
plt.plot(t, y2, 'r',label='Velocidade, y_2(t)')
plt.ylabel('$y_1(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.show()
plt.legend(fontsize=10)

