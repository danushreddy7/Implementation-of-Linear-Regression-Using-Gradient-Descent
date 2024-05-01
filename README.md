# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
    
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))    
```
/*
Program to implement the linear regression using gradient descent.
Developed by:T DANUSH REDDY
RegisterNumber:212223040029  
*/
```

## Output:
# profile prediction:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/e7ca3b79-a2f8-463c-be8c-d51e3ffe7c2b)
# Function:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/2fe52a7b-abcf-4f3c-acc6-f0d7ac72307c)
# Gradient descent:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/2274773f-04f9-4e3a-9276-8aa0e168fb87)
# cost function using gradient descent:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/8253f483-1179-4953-a1c7-35bc33f2c84b)
# linear regression using profile prediction:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/6442bb52-fb11-4814-a8d1-b6009f453fab)
# profile presiction for the population of 35000:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/9f7ba273-3dbc-4619-8629-34d5e15ed8f3)
# profile prediction for the population of 70000:
![image](https://github.com/danushreddy7/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149035740/3a7439f7-2d3a-4bf3-b9ef-f1faa02ac1b9)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
