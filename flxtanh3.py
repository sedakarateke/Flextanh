import numpy as np
import math
import matplotlib.pyplot as plt

def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def tanh_derivative(x):
    return (4*np.exp(-2*x))/((1+np.exp(-2*x))**2)
    
def flxtanh(x,q,a,b):
    return ((a*(1-q*np.e**(-2*b*x))/(1+q*np.e**(-2*b*x))))

def flxtanh_derivative(x,q,a,b):    
    return (4*a*b*q*np.exp(-b*x))/((1 + q*np.exp(-b*x))**2)

def generalized_logistic_type(x, q, b):
    return 1/(1+q*(np.e**(-b*x)))

def generalized_logistic_type_derivative(x, q, b):
    return (b*q)/((np.e**(b*x))*((1+q*np.e**(-b*x))**2))           

def plot_activation_functions(a, b, q):
    x_values = np.linspace(-20, 20, 1000)
    plt.figure(figsize=(15, 10))
    color1=['b','r','g','m']
   
    y_values = tanh(x_values)
    label = 'tanh'
    plt.plot(x_values, y_values, color1[0],label=label)
    
    '''
    y_values = LeCun_tanh(x_values,a, b)
    label = 'LeCun tanh'
    plt.plot(x_values, y_values, color1[1],label=label)
    '''
    
    y_values = flxtanh(x_values, q, a, b)
    label = 'Flexible tanh'
    plt.plot(x_values, y_values, color1[2],label=label)
    
    y_values = generalized_logistic_type(x_values, q, b)
    label = 'Generalized logistic-type'
    plt.plot(x_values, y_values, color1[3],label=label)
    
    
    y_values = tanh_derivative(x_values)
    label = 'tanh derivative'
    plt.plot(x_values, y_values,color1[0]+'--',label=label)
    
    '''
    y_values = LeCun_tanh_derivative(x_values,a, b)
    label = 'LeCun tanh derivative'
    plt.plot(x_values, y_values, color1[1]+'--',label=label)
    '''
    
    y_values = flxtanh_derivative(x_values, q, a, b)
    label = 'Flexible tanh derivative'
    plt.plot(x_values, y_values,color1[2]+'--',label=label)    
    
    y_values = generalized_logistic_type_derivative(x_values, q, b)
    label = 'Generalized logistic-type derivative'
    plt.plot(x_values, y_values,color1[3]+'--',label=label)
   
    #plt.title('Comparisons of tanh, Flexible tanh and \n generalized logistic-type activation functions (a=2, b=1.5, q=5)')
    plt.xlabel('x')
    plt.ylabel('Activation values/Derivatives')
    plt.legend()
    plt.grid(True)
    plt.show()
    

# Different Parameter Values
a=2
b=1.5
q=5
plot_activation_functions(a, b, q)



