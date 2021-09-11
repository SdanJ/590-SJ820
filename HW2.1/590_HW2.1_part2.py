#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:15:20 2021

@author: jinshengdan
"""

##########################
##### HW 2.1
##### ANLY 590 Section 2
##### Shengdan Jin
##########################

import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt
import json


#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']

#UNCOMMENT FOR VARIOUS MODELS
# model_type="logistic"; NFIT=4; xcol=1; ycol=2;
model_type="linear";   NFIT=4; xcol=1; ycol=2; 
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;

#HYPER-PARAM
#OPT_ALGO='BFGS'

#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX 
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS):
		X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
#SIMILAR TO A PD DATAFRAME
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#EXTRACT AGE<18
#if(model_type=="linear"):
#	max_age=18; y=y[x[:]<max_age]; x=x[x[:]<max_age]; 

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD; y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]

xt=x[train_idx]; yt=y[train_idx]
xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#SAVE HISTORY FOR PLOTTING AT THE END
iterations=[]; iterations_val=[];loss_train=[];  loss_val=[]; para_opt = [] ; para_opt_val=[]
iteration=0
iteration_val=0



def loss(p):
	global iterations,iterations_val,loss_train,loss_val,iteration
	#TRAINING LOSS
	yp=model(xt,p) #model predictions for given parameterization p
	training_loss=(np.mean((yp-yt)**2.0))  #MSE

	#VALIDATION LOSS
	#yp=model(xv,p) #model predictions for given parameterization p
	#validation_loss=(np.mean((yp-yv)**2.0))  #MSE

	#WRITE TO SCREEN
	#if(iteration%25==0): print(iteration,training_loss,validation_loss) #,p)
	
	#RECORD FOR PLOTING
	#loss_train.append(training_loss); loss_val.append(validation_loss)
	#iterations.append(iteration)

	#iteration+=1

	return training_loss


## validation loss
def loss_v(p):
    
    #VALIDATION LOSS
    yp=model(xv,p) #model predictions for given parameterization p
    
    validation_loss=(np.mean((yp-yv)**2.0))  #MSE
    
    return validation_loss


po=np.random.uniform(0.5,1.,size=NFIT)

def optimizer(objective, algo='GD', LR=0.001, method='batch'):
    
    if(method == 'batch'):

        #PARAM
        xmin=-50; xmax=50;  
        NDIM=4
        xi=np.random.uniform(xmin,xmax,NDIM) #INITIAL GUESS FOR OPTIMIZEER	
        #PARAM
        dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE
        t=0 	 							#INITIAL ITERATION COUNTER
        tmax=100000							#MAX NUMBER OF ITERATION
        tol=10**-10		                    #EXIT AFTER CHANGE IN F IS LESS THAN THIS 
        v=0
        c=0
        m=0
        
    
        print("INITAL GUESS:",xi)
        
        while(t<=tmax):
            t=t+1
    
        
            	#NUMERICALLY COMPUTE GRADIENT 
            df_dx=np.zeros(NDIM)
            df_dx_val = np.zeros(NDIM)
            
            #### Training Loss
            if(objective == 'training_loss'):
    
                for i in range(0,NDIM):
                		dX=np.zeros(NDIM);
                		dX[i]=dx; 
                		xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
                		df_dx[i]=(loss(xi)-loss(xm1))/dx
                        
                if(algo=='GD'):
                    	#print(xi.shape,df_dx.shape)
                    xip1=xi-LR*df_dx #STEP 
                if(algo=='GD+momentum'):
                    mu=0.5
                    v=mu*v-LR*df_dx
                    xip1 = xi+v
                if(algo=='RMSprop'):
                    c=0.99*c+(1-0.99)*df_dx**2
                    xip1 = xi-LR*df_dx/(np.sqrt(c)+1e-6)
                if(algo=='ADAM'):
                    m=0.9*m+(1-0.9)*df_dx
                    v=0.999*v+(1-0.999)*(df_dx**2)
                    xip1 = xi - LR*m/(np.sqrt(v)+1e-6)

                
                if(t%10==0):
                    df=np.mean(np.absolute(loss(xip1)-loss(xi)))
                    #print(t,"	",xi,"	","	",self.loss(xi)) #,df) 
                    iterations.append(t)
                    para_opt.append(xi)
                    loss_train.append(loss(xi))
        
                		
                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")
                        break
                
        
                
                	#UPDATE FOR NEXT ITERATION OF LOOP	
                xi=xip1
            
            #### Validation Loss
            if(objective == 'validation_loss'):
                 
                 for i in range(0,NDIM):
                     dX=np.zeros(NDIM)
                     dX[i]=dx
                     xm1=xi-dX
                     df_dx_val[i]=(loss_v(xi)-loss_v(xm1))/dx
    
                 if(algo=='GD'):
                     #print(xi.shape,df_dx.shape)
                     xip1_val = xi-LR*df_dx_val #STEP 
                 if(algo=='GD+momentum'):
                     mu=0.5
                     v=mu*v-LR*df_dx_val
                     xip1_val = xi+v
                 if(algo=='RMSprop'):
                     c=0.99*c+(1-0.99)*df_dx_val**2
                     xip1_val = xi-LR*df_dx_val/(np.sqrt(c)+1e-6)
                 if(algo=='ADAM'):
                     m=0.9*m+(1-0.9)*df_dx_val
                     v=0.999*v+(1-0.999)*(df_dx_val**2)
                     xip1_val = xi - LR*m/(np.sqrt(v)+1e-6)
                     
                
                 if(t%10==0):
                     df=np.mean(np.absolute(loss(xip1_val)-loss(xi)))
                     #print(t,"	",xi,"	","	",self.loss_v(xi)) #,df) 
                     iterations_val.append(t)
                     para_opt_val.append(xi)
                     loss_val.append(loss_v(xi))
                		
                     if(df<tol):
                         print("STOPPING CRITERION MET (STOPPING TRAINING)")
                         break
            
    
            
            	     #UPDATE FOR NEXT ITERATION OF LOOP	
                 xi=xip1_val
                 
    



#TRAIN MODEL USING SCIPY MINIMIZ 
#res = minimize(loss, po, method=OPT_ALGO, tol=1e-15)
#popt=res.x
 
alg = ['GD', 'GD+momentum', 'RMSprop', 'ADAM']  

for i in alg:     
    optimizer('training_loss',i,0.0001)
    optimizer('validation_loss',i,0.0001)
    
    popt= para_opt[-1]
    print("OPTIMAL PARAM:",popt)
    
    #PREDICTIONS
    xm=np.array(sorted(xt))
    yp=np.array(model(xm,popt))
    
    #UN-NORMALIZE
    def unnorm_x(x): 
    	return XSTD*x+XMEAN  
    def unnorm_y(y): 
    	return YSTD*y+YMEAN 
    
    
    #FUNCTION PLOTS
    if(IPLOT):
    
        ys = scipy.signal.savgol_filter(unnorm_y(yp), 5, 4)  # window size , polynomial order
        # #QUADRATICALLY INTERPOLATE THE savgol_filter DATA ONTO LESS DENSE MESH 
        xs1=np.linspace(min(unnorm_x(xm)), max(unnorm_x(xm)), int(0.25*len(unnorm_x(xm))))
        F=interp1d(unnorm_x(xm), ys, kind='quadratic')
        ys1=F(xs1)
        
        plt.plot(xs1,ys1,color='r',label = 'Model',linewidth=3.0)
        plt.scatter(unnorm_x(xv),unnorm_y(yv), marker='*', label = 'Validation Set')
        plt.scatter(unnorm_x(xt),unnorm_y(yt), label = 'Training Set')
        plt.title("Comparison Among Model, Training Set and Validation Set -- %s" %i)  # set plot title
        plt.xlabel("X") # set name for x-axis
        plt.ylabel("Y")  # set name for y-axis
        plt.legend(loc="upper right") # set legend position
        plt.show()
    
    #PARITY PLOTS
    if(IPLOT):
      
        ## Parity Plot...
        # Smooth the line
        ys = scipy.signal.savgol_filter(model(xt,popt), 5, 4)
        ys_= scipy.signal.savgol_filter(model(xv,popt), 5, 4)
        # #QUADRATICALLY INTERPOLATE THE savgol_filter DATA ONTO LESS DENSE MESH 
        yv1=np.linspace(min(yt), max(yt), int(0.25*len(yt)))
        F=interp1d(yt, ys)
        ys1=F(yv1) 
       
        yv1_=np.linspace(min(yv), max(yv), int(0.25*len(yv)))
        F_=interp1d(yv, ys_)
        ys1_=F_(yv1_) 
        
        plt.plot(yv1,ys1,'o', label='Training set')
        plt.plot(yv1_,ys1_,'o', label='Validation set')
        plt.title("Parity Plot -- %s" %i)  # set plot title
        plt.xlabel("Ground Truth") # set name for x-axis
        plt.ylabel("Prediction")  # set name for y-axis
        plt.legend(loc="upper right") # set legend position
        plt.show()
    
    #MONITOR TRAINING AND VALIDATION LOSS  
    if(IPLOT):
    
        plt.scatter(iterations, loss_train,label='Training Loss')
        plt.scatter(iterations_val, loss_val, color='r', label='Validation Loss')
        plt.title("Loss by Iterations-- %s" %i)  # set plot title
        plt.xlabel("Optimizer Iteration") # set name for x-axis
        plt.ylabel("Loss")  # set name for y-axis
        plt.legend(loc="upper right") # set legend position
        #plt.ylim(0.0, 1.5)
        plt.show()
	
# I HAVE WORKED THROUGH THIS EXAMPLE AND UNDERSTAND EVERYTHING THAT IT IS DOING

