# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:45:35 2022

@author: valter

Exemplo de simulação numérica de controle
"""

# importa bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


plt.close('all') # Fecha gráficos

"""
Define EDO - Caso 2: tanque não linear

\dot{h}(t) = Fi(t)/A - (\beta/A)*\sqrt(h(t)) 

A = \pi r**2; r = 10cm
""" 

def dhdt(t,h,A,beta,Fi): # argumentos A, beta e Fi precisam ser informados!
    if h<0: h=0
    return Fi/A-(beta/A)*np.sqrt(h)    

# Controle on-off
# 1) vazão de entrada inicialmente nula (Fi = 0) 
#    condição inicial (h(0) = 0,5 m)):

# tempo    
t0 = 0    # tempo inicial
tf = 100  # tempo final
Ts = .1    # tempo entre ações de controle


# parametros do sistema
r = 0.1   # raio do tanque cilindrico
area = np.pi*r**2 # area
beta = .0015     # coeficiente beta

# condição inicial e valor de Fi (=u)
h0 = 0.5 # nivel inicial

"""
Controle:
     A cada intervalo de Ts segundos, os valores de nível 
     são medidos, uma decisão de controle é tomada e aplicada 
     no sistema
     Neste exemplo o controle é ligado em t = t0.
""" 
h_R = 0.45
t = t0
h = h0 # começa a simular com a medicao do nivel atual

u_max = 0.0094 # veja tb com 0.001!
u_min = 0

niveis = np.zeros(int(np.round(tf/Ts)))
controle = np.zeros(np.size(niveis));
tempo = np.linspace(0,len(niveis)-1,len(niveis))*Ts

contador = 0
niveis[contador] = h

while t<(tf-Ts-1e-3):
    # calcula a acao de controle:
    if h < h_R:
       u = u_max
    else:
       u = u_min
    controle[contador] = u   
    # Simula o sistema entre os instantes t e t+Ts     
    sol = solve_ivp(dhdt, t_span=(t,t+Ts), y0=[h], method='RK23',
                 t_eval=[t, t+Ts], args=(area, beta, u))
    
    #h = sol.y[0][-1] # "mede" o valor do nivel
                     # teste com a próxima linha (ruido)
    h = sol.y[0][-1]+ np.random.normal(0,0.002)
    t += Ts          # incrementa o tempo de Ts
    contador += 1
    niveis[contador] = h

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(tempo, niveis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.ylim([0.4, 0.55])
plt.subplot(2,1,2)
plt.plot(tempo, controle, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

