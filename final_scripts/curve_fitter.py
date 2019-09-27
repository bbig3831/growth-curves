import os
import numpy as np
import warnings
from lmfit import Model
import pandas as pd
from final_scripts.get_data import OUTPUT_PATH
from final_scripts.plotters import makeCountrySubplots, makeCaterpillarPlot, makeBICPlot

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
    fittedModel = mod.fit(y, t=t, verbose=False)
    return fittedModel

def storeResults(model):
    """
    Store pertinent information from ModelResult object for fitted growth curve
    :param model: lmfit.Model.ModelResult object
    :return: Dictionary with best fit values, confidence intervals, and BIC
    """
    # Best fit values
    bestVals = model.best_values
    # Confidence intervals
    confLow = {param+'_low':model.conf_interval()[param][1][1] for param in model.model._param_names}
    confHigh = {param+'_high':model.conf_interval()[param][5][1] for param in model.model._param_names}
    return {**bestVals, **confLow, **confHigh, 'BIC': model.bic}

def storeConstants(y):
    """Calculate baseline constants from input data"""
    constants = {
        'currentCoverage': y[0][-1],
        'maxChange': np.diff(y).max(),
        'avgChange': np.diff(y).mean(),
        'obsDelay': np.nonzero(y)[1][0]
    }
    return constants

def saveToExcel(df, file_path):
    """Save result set and Tables 2/3 to Excel"""
    writer = pd.ExcelWriter(file_path)
    df.to_excel(writer, sheet_name='Results')
    # Write Table 2
    table2 = df[['Country Name','avgChange','rank_avgChange','mu_gompertz','rank_mu_gompertz',
                 'mu_logistic','rank_mu_logistic']].sort_values(by=['Country Name'])
    table2.to_excel(writer, sheet_name='Table 2', index=False)
    # Write Table 3
    table3 = df[['Country Name','obsDelay','rank_obsDelay','lamb_gompertz','rank_lamb_gompertz',
                 'lamb_logistic','rank_lamb_logistic']].sort_values(by=['Country Name'])
    table3.to_excel(writer, sheet_name='Table 3', index=False)
    writer.save()


if __name__ == '__main__':
    # Turn off warnings
    warnings.filterwarnings('ignore')

    # Import ART coverage data
    artDf = pd.read_csv(os.path.join(OUTPUT_PATH, 'API_SH.HIV.ARTC.ZS_DS2_EN_csv_v2_116032.csv'), header=2)
    artDf.drop(labels=['Indicator Name', 'Indicator Code', 'Unnamed: 22'], axis=1, inplace=True)

    # Get countries with >= 30% ART coverage
    highARTCountries = artDf[artDf['2017']>=30]['Country Code'].tolist()

    # Import population coverage data
    popDf = pd.read_csv(os.path.join(OUTPUT_PATH, 'API_SP.POP.TOTL_DS2_EN_csv_v2_116033.csv'), header=2)
    popList = popDf[popDf['2017']>=1e6]['Country Code'].tolist()

    # Import data to get countries in sub-Saharan Africa (World Bank GroupCode = SSF)
    classDf = pd.read_excel(os.path.join(OUTPUT_PATH, 'WB_Classification.xls'), sheet_name='Groups')
    ssfCountries = classDf[classDf['GroupCode']=='SSF']['CountryCode'].tolist()

    # Import HIV data to get countries with max prevalence >= 5%
    hivDf = pd.read_csv(os.path.join(OUTPUT_PATH, 'API_SH.DYN.AIDS.ZS_DS2_EN_csv_v2_150659.csv'), header=2)
    hivDf['max_HIV_prevalence'] = hivDf[[col for col in hivDf if col.startswith('20')]].max(axis=1)
    highHIVCountries = hivDf[hivDf['max_HIV_prevalence']>=5]['Country Code'].tolist()

    # Set list of countries based on restrictions
    countryList = list(set(popList) & set(ssfCountries))
    countryList.sort()

    # Define subgroup based on HIV prevalence/ART coverage restrictions
    subgroupList = list(set(popList) & set(ssfCountries) & set(highARTCountries) & set(highHIVCountries))
    subgroupList.sort()

    resultsDict = {'logistic':{}, 'gompertz':{}, 'constants':{}}
    modelDict = {}
    for country in countryList:
        print(f'Fitting curves for {country}')
        df = artDf[artDf['Country Code'] == country]
        # Get time values
        tStr = [col for col in df if col.startswith('20')]
        # Center around 2000
        tInt = [int(t) - 2000 for t in tStr]
        # Get coverage values
        y = np.asarray(df[tStr])
        if np.isnan(np.sum(y)):
            print(f'Skipping {country}, contains NaN values.')
            continue
        # Fit logistic curve
        logisticFit = curveFitter(function=logistic, y=y, t=tInt)
        # Fit Gompertz curve
        gompertzFit = curveFitter(function=gompertz, y=y, t=tInt)
        if country in ['BWA','GAB','RWA','KEN','UGA','ZAF']:
            modelDict[country] = {'logistic': logisticFit, 'gompertz': gompertzFit}
        # Store results
        resultsDict['logistic'][country] = storeResults(logisticFit)
        resultsDict['gompertz'][country] = storeResults(gompertzFit)
        resultsDict['constants'][country] = storeConstants(y)

    # Construct dataframe with results
    dfDict = {func: pd.DataFrame.from_dict(resultsDict[func], orient='index') for func in resultsDict.keys()}
    joinedDf = pd.merge(left=dfDict['logistic'], right=dfDict['gompertz'], how='left',
                        left_index=True, right_index=True, suffixes=('_logistic', '_gompertz'))
    joinedDf = pd.merge(left=joinedDf, right=dfDict['constants'], how='left',
                        left_index=True, right_index=True)
    joinedDf['delta_BIC'] = joinedDf['BIC_gompertz']-joinedDf['BIC_logistic']
    joinedDf = pd.merge(left=joinedDf, right=artDf[['Country Code','Country Name']], how='left',
                        left_index=True, right_on='Country Code')

    # Add rankings based on subset of metrics
    for metric in ['avgChange','mu_gompertz','mu_logistic']:
        joinedDf['rank_'+metric] = joinedDf[metric].rank(ascending=False, method='min')

    for metric in ['obsDelay', 'lamb_gompertz', 'lamb_logistic']:
        joinedDf['rank_'+metric] = joinedDf[metric].rank(ascending=True, method='min')

    # Create subgroup dataframe
    subgroupDf = joinedDf[joinedDf['Country Code'].isin(subgroupList)]

    # Write output to Excel
    print('Saving results to Excel')
    saveToExcel(joinedDf, '/home/ben/Desktop/ART_model_results.xlsx')
    saveToExcel(subgroupDf, '/home/ben/Desktop/ART_model_results_subgroup.xlsx')

    # Make plots
    print('Making figures for manuscript')
    makeCountrySubplots(artDf=artDf, countries=['BWA', 'GAB', 'RWA', 'KEN', 'UGA', 'ZAF'],
                        model_dict=modelDict, output_path='/home/ben/Desktop/countries.png')

    # Make caterpillar plots
    plotDict = {'all':joinedDf, 'subgroup':subgroupDf}
    for label, df in plotDict.items():
        makeCaterpillarPlot(df=df,
                            metric_dict={'mu_gompertz':'$\mu$ - Gompertz', 'mu_logistic':'$\mu$ - Logistic',
                                         'avgChange':'Avg. change'},
                            xlabel='Rate of change (percentage points/year)',
                            output_path=f'/home/ben/Desktop/mu_plot_{label}.png',
                            sort_list=[True, False])
        makeCaterpillarPlot(df=df,
                            metric_dict={'lamb_gompertz':'$\lambda$ - Gompertz', 'lamb_logistic':'$\lambda$ - Logistic',
                                         'obsDelay':'Observed'},
                            xlabel='Delay in scale-up (years)', output_path=f'/home/ben/Desktop/lamb_plot_{label}.png')

        # Make BIC plot
        makeBICPlot(df=df, output_path=f'/home/ben/Desktop/bic_plot_{label}.png')