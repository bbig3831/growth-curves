import os
import numpy as np
from lmfit import Model
import pandas as pd
from final_scripts.get_data import OUTPUT_PATH

def logistic(t,A,mu,lamb):
    """
    Logistic growth curve function
    :param t: Independent variable (time)
    :param A: Carrying capacity
    :param mu: Maximum growth rate
    :param lamb: Lag time prior to the start of scale-up
    :return: Value of the dependent variable
    """
    y = A/(1+np.exp((4*mu/A)*(lamb-t)+2))
    return y

def gompertz(t,A,mu,lamb):
    """
    Gompertz growth curve function
    :param t: Independent variable (time)
    :param A: Carrying capacity
    :param mu: Maximum growth rate
    :param lamb: Lag time prior to the start of scale-up
    :return: Value of the dependent variable
    """
    y = A*np.exp(-np.exp(mu*np.exp(1)/A*(lamb-t)+1))
    return y

def curveFitter(function, y, t):
    """
    Fits a growth curve model based on the given inputs and parameterized functions
    :param function: Growth curve function object to be passed to lmfit.Model()
    :param y: Dependent variable values
    :param t: Independent variable (time) values
    :return: lmfit.model.ModelResult object
    """
    # Initialize model
    mod = Model(function)
    # Set model hints
    mod.set_param_hint('A', value=y.max(), min=0, max=100)
    mod.set_param_hint('mu', value=np.diff(y).max(), min=0)
    mod.set_param_hint('lamb', value=3)
    # Fit model
    fittedModel = mod.fit(y, t=t, verbose=True)
    return fittedModel

def oldCurveFitter(function, df, t, country):
    y = np.asarray(df[country])
    mod = Model(function)
    ### Set parameter hints
    #A - Restrict coverage between 0 and 100%, guess current maximum value
    mod.set_param_hint('A',value=y.max(),min=0,max=100)
    #Mu - Rate of change, guess max difference
    mod.set_param_hint('mu', value=df[country].diff(1).max())
    #Lambda - Lag time, guess 3
    mod.set_param_hint('lamb',value=3)
    ### Fit model
    result = mod.fit(y,t=t,verbose=False)
    return result

# TODO: Add function to store output/results

if __name__ == '__main__':
    # TODO: Add for loop to run curve fitter for each country
    # Import ART coverage data
    artDf = pd.read_csv(os.path.join(OUTPUT_PATH, 'API_SH.HIV.ARTC.ZS_DS2_EN_csv_v2_111510.csv'), header=2)
    artDf.drop(labels=['Indicator Name', 'Indicator Code', 'Unnamed: 23'], axis=1, inplace=True)
    # Import population coverage data
    popDf = pd.read_csv(os.path.join(OUTPUT_PATH, 'API_SP.POP.TOTL_DS2_EN_csv_v2_111511.csv'), header=2)
    popDf.drop(labels=['Indicator Name', 'Indicator Code', 'Unnamed: 23'], axis=1, inplace=True)
    df = artDf[artDf['Country Code'] == 'BWA'].drop(labels=['2018'], axis=1)
    # Get time values
    tStr = [col for col in df if col.startswith('20')]
    # Center around 2000
    tInt = [int(t)-2000 for t in tStr]
    # Get coverage values
    y = np.asarray(df[tStr])
    fit = curveFitter(function=logistic, y=y, t=tInt)
    tEval = np.asarray(range(0,18))
