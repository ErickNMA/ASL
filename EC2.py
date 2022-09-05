from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.misc import derivative
import matplotlib.pyplot as plt
import numpy as np
import math as m

#Derivada da função 1:
def df1(t, h, A, alfa, Fi):
    if(h < 0):
        h = 0
    return ((Fi/A) - ((alfa/A)*h))

#Derivada da função 2:
def df2(t, h, A, beta, Fi):
    if(h < 0):
        h = 0
    return ((Fi/A) - ((beta/A)*m.sqrt(h)))

#Erro:
def erro(t, x_d, x=None):
    if(x is None):
        x = x_d
        x_d = t
    return (x_d-x)

#Parâmetros do sistema:
area = .2





##########################################################################################################################

##########################################################################################################################

#Item 1:

alfa = .4
beta = .4

#Condições iniciais:
h0 = 1
u = 0.04

#Caso 1:

#Parâmetros de simulação:
t = 0
tf = 3
step = .01

#Looping:
h = h0
h_axis = []
h_axis.append(h)
u_axis = []
u_axis.append(u)
t_axis = []
t_axis.append(t)

while(t < tf):
    #Controle:
    u_axis.append(u)

    #Solução da EDO:
    sol = solve_ivp(df1, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, alfa, u))

    h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(1)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#Caso 2:

#Condições iniciais:
u = 0.04

#Parâmetros de simulação:
t = 0
tf = 3
step = .01

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    #Controle:
    u_axis.append(u)

    #Solução da EDO:
    sol = solve_ivp(df2, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

    h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(2)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

##########################################################################################################################

##########################################################################################################################

#Item 2:

alfa = .4
beta = .3

#Condições iniciais:
h0 = 0.5
u = 0.04

#Altura desejada (ponto de equilíbrio):
h_D = 0.6

#Caso 1:

#Parâmetros de simulação:
t = 0
tf = 3
step = .01

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    #Controle:
    if(h < h_D):
        u = 1.2
    else:
        u = 0.2
    u_axis.append(u)

    #Solução da EDO:
    sol = solve_ivp(df1, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, alfa, u))

    h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(3)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#Caso 2:

#Condições iniciais:
u = 0.04

#Parâmetros de simulação:
t = 0
tf = 3
step = .01

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    #Controle:
    if(h < h_D):
        u = 1.2
    else:
        u = 0.2
    u_axis.append(u)

    #Solução da EDO:
    sol = solve_ivp(df2, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

    h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(4)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

##########################################################################################################################

##########################################################################################################################

#Item 3:

alfa = .1
beta = .1

#Condições iniciais:
h0 = 0.5
u = 0.04

#Altura desejada (ponto de equilíbrio):
h_D = 0.6

#Caso 1:

#Parâmetros de simulação:
t = 0
tf = 2
step = .01

#Constantes do controlador PI:
Kp = 1
Ki = 1

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    #Controle:
    #Proporcional:
    prop = erro(h_D, h)
    #Integrativo:
    integration, err = quad(erro, t, t+step, args=(h_D, h))
    u = ((Kp*prop) + (Ki*integration))
    if(u < 0):
        u = 0
    u_axis.append(u)

    #Solução da EDO:
    sol = solve_ivp(df1, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, alfa, u))

    h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(5)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

#Caso 2:

#Parâmetros de simulação:
t = 0
tf = 2
step = .01

#Constantes do controlador PI:
Kp = 1
Ki = 1

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    #Controle:
    #Proporcional:
    prop = erro(h_D, h)
    #Integrativo:
    integration, err = quad(erro, t, t+step, args=(h_D, h))
    u = ((Kp*prop) + (Ki*integration))
    if(u < 0):
        u = 0
    u_axis.append(u)

    #Solução da EDO:
    sol = solve_ivp(df2, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

    h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(6)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

##########################################################################################################################

##########################################################################################################################

# Item 5:

# on-off:

alfa = .4
beta = .3

#Condições iniciais:
u = 0.04

#Parâmetros de simulação:
t = 0
tf = 150
step = .01

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    if((t > (50-1e-6)) and (t < (50+1e-6))):
        h = 0.4
    elif((t > (100-1e-6)) and (t < (100+1e-6))):
        h = 0.8
    else:
        #Controle:
        if(h < h_D):
            u = 1.2
        else:
            u = 0.2

        #Solução da EDO:
        sol = solve_ivp(df2, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

        h = sol.y[0][-1] #+ np.random.normal(0, 0.002)
    
    

    u_axis.append(u)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(7)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()

# PI:

alfa = .1
beta = .1

#Condições iniciais:
u = 0.04

#Parâmetros de simulação:
t = 0
tf = 150
step = .01

#Constantes do controlador PI:
Kp = 50
Ki = 50

#Looping:
h = h0
h_axis.clear()
h_axis.append(h)
u_axis.clear()
u_axis.append(u)
t_axis.clear()
t_axis.append(t)

while(t < tf):
    if((t > (50-1e-6)) and (t < (50+1e-6))):
        h = 0.4
    elif((t > (100-1e-6)) and (t < (100+1e-6))):
        h = 0.8
    else:
        #Controle:
        #Proporcional:
        prop = erro(h_D, h)
        #Integrativo:
        integration, err = quad(erro, t, t+step, args=(h_D, h))
        u = ((Kp*prop) + (Ki*integration))
        if(u < 0):
            u = 0

        #Solução da EDO:
        sol = solve_ivp(df2, t_span=(t, t+step), y0=[h], method='RK23', t_eval=[t, t+step], args=(area, beta, u))

        h = sol.y[0][-1] #+ np.random.normal(0, 0.002)    

    u_axis.append(u)
    h_axis.append(h)
    t += step
    t_axis.append(t)

#Plotagem do gráfico:
plt.figure(8)
plt.subplot(2,1,1)
#plt.ylim([0.74, 0.76])
plt.plot(t_axis, h_axis, 'k',label='Nível, h(t)')
plt.ylabel('$h(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.subplot(2,1,2)
#plt.ylim([-0.02, 0.02])
plt.plot(t_axis, u_axis, 'b',label='Controle, u(t)')
plt.ylabel('$u(t)$', fontsize=12)
plt.xlabel('$t$', fontsize=12)
plt.grid(axis='both')
plt.legend(fontsize=10)
plt.show()