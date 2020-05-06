#Lea Setruk 345226179
#Aviva Shneor Simchon 317766731
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sts
import scipy.stats as stats
import requests
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.stats import lognorm
import seaborn as sns
from scipy.optimize import minimize
import math
import numpy as np
import copy
import sys



file = sys.argv[1]
data = pd.read_csv(file)

#data = pd.read_csv(r'D:\OPTI\baltimore-city-employee-salaries-fy2019-1.csv')
#data = pd.read_csv(r'C:\Users\utilisateur\Documents\Master 1\Optimisation\baltimore-city-employee-salaries-fy2019-1.csv')


copy_data = data.copy()

data.head()


null_values_col = copy_data.isnull().sum()
null_values_col = null_values_col[null_values_col != 0].sort_values(ascending = False).reset_index()
null_values_col.columns = ["variable", "number of missing"]
null_values_col.head()


copy_data = copy_data.dropna()


copy_data.isnull().any()


copy_data.columns


data_for_gmm = np.log(copy_data[['Gross']])
data_for_gmm= data_for_gmm.dropna().values
data_for_gmm = data_for_gmm [data_for_gmm != -np.inf]
data_for_gmm = data_for_gmm [data_for_gmm != np.inf]
np.isnan(data_for_gmm).sum()
data_for_gmm=data_for_gmm.reshape(-1,1)




#GMM
gm_aic = []
gm_bic= []
for i in range(2,7):
    gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(data_for_gmm)
    gaussian_mixture_model_labels = gm.fit_predict(data_for_gmm)
    gm_aic.append(gm.aic(data_for_gmm))   


plt.figure(figsize=(6,3))
plt.title("The Gaussian Mixture model AIC \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,7)],y=np.log(gm_aic),s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Log of Gaussian mixture AIC score",fontsize=15)
plt.xticks([i for i in range(2,7)],fontsize=14)
plt.yticks(fontsize=12)
plt.show()



copy_data['Gross'].describe()



# ax = sns.distplot(copy_data['Gross'],
#                   bins=100,
#                   kde=True,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Distribution', ylabel='Frequency')





col = copy_data['Gross']
col.head()




data_more_25K= col[(col > 25000)]
data_log_more_25K =  np.log(data_more_25K)
#col[col<1]
data_log_more_25K = data_log_more_25K [data_log_more_25K != -np.inf]



data_less_25K= col[(col < 25000)]
data_log_less_25K =  np.log(data_less_25K)
#col[col<1]
data_log_less_25K = data_log_less_25K [data_log_less_25K != -np.inf]



print(np.mean(data_less_25K))
print(np.std(data_less_25K))
print(np.mean(data_more_25K))
print(np.std(data_more_25K))



print(np.mean(data_log_less_25K))
print(np.std(data_log_less_25K))
print(np.mean(data_log_more_25K))
print(np.std(data_log_more_25K))


print(sum(np.isnan(data_log_less_25K)))



#def lognormal(mu,sigma,x):
    #return np.exp(-(np.log(x)-mu)**2/(2*sigma**2))/(x*sigma*np.sqrt(2*np.pi))



#Lognormal density

def func1(mu,sigma,dataX):
    return 1/(dataX*sigma*np.sqrt(np.pi*2))*np.exp(-(np.log(dataX)-mu)**2/(2*sigma**2))



def theoretical_function(a,mu1,sigma1,mu2,sigma2,dataX):
    a=max(min(a,1),0)
    return a*func1(mu1,sigma1,dataX) + (1-a)*func1(mu2,sigma2,dataX) 



def theory_for_3(a,b,mu1,sigma1,mu2,sigma2, mu3, sigma3, dataX):
    a=max(min(a,1),0)
    b=min(b,1-a)
    return a*func1(mu1,sigma1,dataX) + b*func1(mu2,sigma2,dataX) + (1-a-b)*func1(mu3, sigma3, dataX)


def Empiric(col, lnspc):
    digitized = np.digitize(col, lnspc)
    bin_count = [np.sum(np.unique(col[digitized == i],return_counts=True)[1]) for i in range(1, len(lnspc)+1)]
    return np.array(bin_count)



def compute_distance(y_empiric,y_teoretic):
    return stats.ks_2samp(y_empiric,y_teoretic)



def costfunc(params):
    print(params)
    a,mu1,sigma1,mu2,sigma2 = params
    x= np.linspace(1,280000,100)
    y_Emp =Empiric(col, x)
    y_teo =theoretical_function(a,mu1,sigma1,mu2,sigma2,x)
    #Normalization
    y_Emp = y_Emp/sum(y_Emp)
    y_teo = y_teo/sum(y_teo)
    #return sum((y_teo - y_Emp)**2)
    #return np.max(abs(y_teo-y_Emp))
    stat,p_val=compute_distance(y_Emp ,y_teo)
    return stat #compute_distance(y_Emp ,y_teo)



#from scipy.optimize import minimize
params_for_teo2 = minimize(costfunc,( 0.3 , 11.4  ,  2.55625, 10.8, 0.5),method = 'Nelder-Mead') 




params_for_teo2 = params_for_teo2.x.tolist()
print(params_for_teo2)



def costfunc_for_3(params):
    a,b,mu1,sigma1,mu2,sigma2, mu3, sigma3 = params
    x= np.linspace(1,280000,100)
    y_Emp =Empiric(col, x)
    y_teo3=theory_for_3(a,b,mu1,sigma1,mu2,sigma2, mu3, sigma3, x)
    
    #normalization
    y_Emp = y_Emp/sum(y_Emp)
    y_teo3= y_teo3/sum(y_teo3)

    return np.max(abs(np.cumsum(y_teo3)-np.cumsum(y_Emp)))




minimize(costfunc_for_3,
         (0.06320954,  0.11235451, 14.87070052,  3.93917328,  8.90019041,0.87787672, 11.03283469,  0.48644798),
         method = 'Nelder-Mead')


'''
    Nelder-Mead algorithm.
'''


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=100,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print ('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


    
    



obj = nelder_mead(costfunc_for_3, np.array( [0.0625324 ,  0.11375814, 15.25834458,  4.17751721,  9.17601211,0.83649325, 10.84676469,  0.4481840]))
arr ,score = obj
print(arr)



nelder_mead(costfunc, np.array( [ 0.3 , 11.4 , 2.55625, 10.8, 0.5 ]))



#Graph of theoretical density for linear combination with 2 distributions
x= np.linspace(1,280000,100)
#MINIMIZE
y_teo = theoretical_function(0.3084  , 10.9212  ,  2.627825, 10.584   ,  0.514  , x)##MINIMIZE
#y_teo = theoretical_function( 0.3084  , 10.9212  ,  2.627825, 10.584   ,  0.514      , x) #NELDER
#y_teo = theoretical_function( 0.0806048 , 10.98139616,  2.71106116, 10.62538016,  0.58474656, x)
print(sum(y_teo)*np.diff(x)[0])
#print(np.diff(x)[0])

#print(func1(1,np.mean(x),np.std(x),x))
#plt.yscale('log')
plt.plot(x,y_teo/sum(y_teo))

plt.plot(x,y/sum(y))





# NELDER_OUR Graph of theoretical density for linear combination with 2 distributions
x= np.linspace(1,280000,100)
y_teo = theoretical_function( 0.0806048 , 10.98139616,  2.71106116, 10.62538016,  0.58474656, x)
print(sum(y_teo)*np.diff(x)[0])
plt.plot(x,y_teo/sum(y_teo))
plt.plot(x,y/sum(y))




#Graph of theoretical density for linear combination with 2 distributions
y_teo3 = theory_for_3(0.05548718,  0.11041845, 14.76812941,  3.96034319,  9.04263846,
        0.78208364, 11.03297898,  0.48725869, x)
##y_teo3 = theory_for_3( 0.06320954,  0.11235451, 14.87070052,  3.93917328,  8.90019041,0.87787672, 11.03283469,  0.48644798, x)
#y_teo3 = theory_for_3(0.06259244,  0.11673331, 15.24122082,  4.07709043,  9.05073134,0.93954711, 11.03516595,  0.48608989,x)
print(sum(y_teo3)*np.diff(x)[0])
y1=y_teo3/sum(y_teo3)
y2=y/sum(y)
plt.plot(x,y1)
plt.plot(x,y2)
#plt.plot(x,np.cumsum(y1))
#plt.plot(x,np.cumsum(y2))
print(compute_distance(y1,y2))
print(np.max(abs(np.cumsum(y1)-np.cumsum(y2))))



#Plot with our NELDER MEAD
y_teo3 = theory_for_3(0.06320954,  0.11235451, 14.87070052,  3.93917328,  8.90019041,
        0.87787672, 11.03283469,  0.48644798  , x)
print(sum(y_teo3)*np.diff(x)[0])
y1=y_teo3/sum(y_teo3)
y2=y/sum(y)
plt.plot(x,y1)
plt.plot(x,y2)

print(compute_distance(y1,y2))
print(np.max(abs(np.cumsum(y1)-np.cumsum(y2))))


x = np.linspace(1, 280000, 100)
y=Empiric(col, lnspc)
print(sum(y))
plt.plot(x,y/sum(y))
print(sum(y))



def f_2(x):
    return theoretical_function(0.3084  , 10.9212  ,  2.627825, 10.584   ,  0.514  , x)
max_x_for_2 = sp.optimize.fmin(lambda x: -f_2(x), 0)
print(max_x_for_2)




def max_likelihood_2(value_max):
    return theoretical_function(0.3084  , 10.9212  ,  2.627825, 10.584   ,  0.514  , value_max)
max_like_for_2 = max_likelihood_2(max_x_for_2)




def f_3(x):
    return theory_for_3(0.05548718,  0.11041845, 14.76812941,  3.96034319,  9.04263846,
        0.78208364, 11.03297898,  0.48725869, x)
max_x_for_3 = sp.optimize.fmin(lambda x: -f_3(x), 0)
print(max_x_for_3)




def max_likelihood_3(value_max):
    return theory_for_3(0.05548718,  0.11041845, 14.76812941,  3.96034319,  9.04263846,
        0.78208364, 11.03297898,  0.48725869, value_max)
max_like_for_3 = max_likelihood_2(max_x_for_3)



def aic_score_2(nb_params, max_lik):
    aic = 2* nb_params - 2 * np.log(max_likelihood_2(max_lik))
    return aic
aic_for_2 = aic_score_2(5, max_like_for_2)
print(aic_for_2)




def aic_score_3(nb_params, max_lik):
    aic = 2* nb_params - 2 * np.log(max_likelihood_3(max_lik))
    return aic
aic_for_3 = aic_score_3(8, max_like_for_3)
print(aic_for_3)




#plt.plot(x,y_teo)

histo = col.hist(bins=100)
plt.title("Salaries in Baltimore city in 2019 Histogram")
plt.xlabel("Salary($)")
plt.ylabel("Frequency")






#histo = copy_data.Gross.hist(bins=20, alpha=0.5)
histo = col.hist(bins=100)
plt.title("Salaries in Baltimore city in 2019 Histogram")
plt.xlabel("Salary($)")
plt.ylabel("Frequency")


# ## Find distribution and parameters




ax = sns.distplot(col,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Distribution', ylabel='Frequency')




bins = 100
y, x = np.histogram(col, bins=bins, density=True)
#y_log = np.log(y)
# Milieu de chaque classe
x = (x + np.roll(x, -1))[:-1] / 2.0
plt.plot(x,y)




distribution_names = ['norm', 'beta','gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2']

sse = np.inf
sse_thr = 0.10

# for every distribution
for name in distribution_names:

    # Model
	distribution = getattr(sp.stats, name)
	param = distribution.fit(col)

	# Params
	loc = param[-2]
	scale = param[-1]
	arg = param[:-2]

	# PDF
	pdf = distribution.pdf(x, *arg, loc=loc, scale=scale)
	# SSE
	model_sse = np.sum((y - pdf)**2)

	# Taking the smallest sse
	if model_sse < sse :
		best_pdf = pdf
		sse = model_sse
		best_loc = loc
		best_scale = scale
		best_arg = arg
		best_name = name

	# Si en dessous du seuil, quitter la boucle
	if model_sse < sse_thr :
		break







