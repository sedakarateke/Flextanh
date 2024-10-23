import numpy as np
import matplotlib.pyplot as plt

def flxtanh(x,q,a,b):
    return ((a*(1-q*np.e**(-2*b*x))/(1+q*np.e**(-2*b*x))))

def flxtanh_derivative(x,q,a,b):    
    return (4*a*b*q*np.exp(-b*x))/((1 + q*np.exp(-b*x))**2)
        

def plot_flxtanh(a_values, q_values, b_values):
    x_values = np.linspace(-20, 20, 1000)
    plt.figure(figsize=(15, 10))
    color1=['b','r','g','m']

    for a in a_values:
        for b in b_values:
            for q in q_values:
                y_values = flxtanh(x_values, q, a, b)
                label = f'a={a}, b={b}, q={q}'
                plt.plot(x_values, y_values, color1[q_values.index(q)],label=label)
    
    for a in a_values:
        for b in b_values:
            for q in q_values:
                y_values = flxtanh_derivative(x_values, q, a, b)
                #label = f'a={a}, b={b}, q={q}'
                plt.plot(x_values, y_values,color1[q_values.index(q)]+'--')

    #plt.title('flxtanhs and their derivatives for different values of q')
    plt.xlabel('x')
    plt.ylabel('flxtanhs/Derivatives')
    plt.legend()
    plt.grid(True)
    plt.show()
    

# Different Parameter Values
a_values = [2]
b_values =[1.5]
q_values = [0.01, 1, 5, 20]
plot_flxtanh(a_values, q_values, b_values)




