#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 13:10:28 2021

@author: jinshengdan
"""
##########################
##### Assignment 1
##### ANLY 590 Section 2
##### Shengdan Jin
##########################

# Import necessary packages
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import scipy.signal
import warnings

#supress all warnings
warnings.filterwarnings("ignore") 


# Create a class named 'Data' to read in file, partition the data and make visualizations.
class Data():
    def __init__(self,file,x_v,y_v,model_type):
        ### Read in data
        df = pd.read_json(file)
        
        self.model_type = model_type
        
        if model_type == 'linear':
            self.NFIT = 2
            df = df[df['x'] < 18] # for linear regression, fit to age < 18
        else:
            self.NFIT = 4
            
        # x variable
        x =df[x_v]
        # y variable
        y =df[y_v]

        
        ### Normalize input and outout (x and y)
        x_mean = sum(x)/len(x)
        x_std = np.std(x)
        norm_x = (x-x_mean)/x_std
        
        y_mean = sum(y)/len(y)
        y_std = np.std(y)
        norm_y = (y-y_mean)/y_std
        
        
        
        ### Split data into 80% training and 20% validation
        ### Set random_state to save the split 
        self.x_train,self.x_val,self.y_train,self.y_val=train_test_split(norm_x,norm_y,train_size=0.8)#,random_state=820) 
        
        ### Optimize
        #RANDOM INITIAL GUESS FOR FITTING PARAMETERS
        po=np.random.uniform(0.5,1.,size=self.NFIT)
               
        #TRAIN MODEL USING SCIPY OPTIMIZER         
        res = minimize(self.loss, po, method='Nelder-Mead', tol=1e-15)
      
        popt=res.x
        
        print("OPTIMAL PARAM:",popt)
        
        
        
        # PLOT FOR LOSS BY ITERATIONS
        
        self.iteration=0
        self.iteration_val=0
        self.iterations=[]
        self.iterations_val=[]
        self.loss_train=[]
        self.loss_val=[]
        
        ls = minimize(self.loss, po, method='Nelder-Mead', tol=1e-6, callback=self.his_loss)
        ls_val = minimize(self.loss_v, po, method='Nelder-Mead', tol=1e-6, callback=self.his_loss_val)

        
        plt.scatter(self.iterations, self.loss_train,label='Training Loss')
        plt.scatter(self.iterations_val, self.loss_val, color='r', label='Validation Loss')
        plt.title("Loss by Iterations")  # set plot title
        plt.xlabel("Optimizer Iteration") # set name for x-axis
        plt.ylabel("Loss")  # set name for y-axis
        plt.legend(loc="upper right") # set legend position
        #plt.ylim(0.0, 1.5)
        plt.show()
        
        
        
        ## Denormalize
        norm_pred = self.model(self.x_val,popt) 
        pred = (norm_pred*y_std+y_mean).to_numpy() # add to_numpy to change into array to avoid warning
        
        de_x_val = (self.x_val*x_std+x_mean).to_numpy()
        
        de_y_val = (self.y_val*y_std+y_mean).to_numpy()
        de_x_train =(self.x_train*x_std+x_mean).to_numpy()
        de_y_train = (self.y_train*y_std+y_mean).to_numpy() 
        
        
        
        ## Plot Comparison Among Model, Training Set and Validarion Set
        # Smooth the line
        ys = scipy.signal.savgol_filter(pred, 5, 4)  # window size , polynomial order
        # #QUADRATICALLY INTERPOLATE THE savgol_filter DATA ONTO LESS DENSE MESH 
        xs1=np.linspace(min(de_x_val), max(de_x_val), int(0.25*len(de_x_val)))
        F=interp1d(de_x_val, ys, kind='quadratic')
        ys1=F(xs1)
        
        plt.plot(xs1,ys1,color='r',label = 'Model',linewidth=3.0)
        plt.scatter(de_x_val,de_y_val, marker='*', label = 'Validation Set')
        plt.scatter(de_x_train,de_y_train, label = 'Training Set')
        plt.title("Comparison Among Model, Training Set and Validation Set")  # set plot title
        plt.xlabel("X") # set name for x-axis
        plt.ylabel("Y")  # set name for y-axis
        plt.legend(loc="upper right") # set legend position
        plt.show()
        
        
        
        ## Parity Plot...
        # Smooth the line
        ys = scipy.signal.savgol_filter(pred, 5, 4)
        # #QUADRATICALLY INTERPOLATE THE savgol_filter DATA ONTO LESS DENSE MESH 
        yv1=np.linspace(min(de_y_val), max(de_y_val), int(0.25*len(de_y_val)))
        F=interp1d(de_y_val, ys)
        ys1=F(yv1) 
       
        plt.plot(yv1,ys1,color = 'orange')
        plt.title("Parity Plot")  # set plot title
        plt.xlabel("Ground Truth") # set name for x-axis
        plt.ylabel("Prediction")  # set name for y-axis
        plt.show()

        
        
    # Build model 
    ## linear / logistic function    
    def model(self,x,p):
        if self.model_type == 'linear':
            return p[0]*x+p[1]
        if self.model_type == 'logistic':
            return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))
    

    # Loss - MSE
    ## training loss    
    def loss(self,p):
        yp=[]
        for x in self.x_train:
            yp.append(self.model(x,p))
        mse = (sum((np.array(self.y_train) - yp)**2))/len(self.x_train)
        
        return mse
    
    ## validation loss
    def loss_v(self,p):
        yp=[]
        for x in self.x_val:
            yp.append(self.model(x,p))
        mse = (sum((np.array(self.y_val) - yp)**2))/len(self.x_val)
        
        return mse
            
    # Save history for plotting at the end
    ## history for training loss by each iteration
    def his_loss(self,x):
        a = self.loss(x)
        self.loss_train.append(a)
        self.iterations.append(self.iteration)
        self.iteration += 1
    ## history for validation loss by each iteration
    def his_loss_val(self,x):
        a = self.loss_v(x)
        self.loss_val.append(a)
        self.iterations_val.append(self.iteration_val)
        self.iteration_val += 1
        
    

def main():
    #using class methods
    obj1 = Data('/Users/jinshengdan/590-CODES/DATA/weight.json','x','y','linear')
    obj2 = Data('/Users/jinshengdan/590-CODES/DATA/weight.json','x','y','logistic')
    obj3 = Data('/Users/jinshengdan/590-CODES/DATA/weight.json','y','is_adult','logistic')
    #obj.vis_1()
    #obj.method_2()
    
if __name__ == '__main__':
    main()
 
