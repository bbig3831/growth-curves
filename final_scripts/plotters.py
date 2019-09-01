import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def makeCountrySubplots(artDf, countries, model_dict, output_path, show_plot=False):
    """
    Function to make Figure 2, a 3x2 subplot of country-level regressions
    :param artDf: Pandas dataframe with ART coverage data
    :param countries: List of 6 countries to plot
    :param model_dict: Dictionary of lmfit.Model.ModelResult objects
    :param output_path: Output path for PNG of subplots
    :param show_plot: Show plot if true (useful for debugging)
    :return:
    """
    sns.set()
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='none', figsize=(13, 15))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    sns.set_style('darkgrid')
    t2 = np.asarray(range(0, 26))

    for ax, country in zip(axes, countries):
        # ART coverage data
        df = artDf[artDf['Country Code'] == country]
        # Get time values
        t = [int(col)-2000 for col in df if col.startswith('20')]
        tStr = [col for col in df if col.startswith('20')]
        y = df[tStr].values.tolist()[0]
        ax.plot(t, y, 'bs')
        ax.plot(t2, model_dict[country]['gompertz'].eval(t=t2), 'r-')
        ax.plot(t2, model_dict[country]['logistic'].eval(t=t2), 'g--')
        ax.set_xlim([0, 20])
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end + 1, 5))
        ax.xaxis.set_ticklabels(['2000', '2005', '2010', '2015', '2020'], fontsize=12)
        ax.set_xlabel(df['Country Name'].iloc[0], fontsize=13)
        if ax in [ax1, ax3, ax5]:
            ax.set_ylabel('% ART Coverage', fontsize=13)
        plt.rc('ytick', labelsize=12)

    ax4.legend(['Data', 'Gompertz', 'Logistic'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=13)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
    if show_plot:
        plt.show()
    # Clear figure
    plt.clf()

# Add function for making caterpillar plots
def makeCaterpillarPlot(df, metric_dict, xlabel, output_path, sort_list=[True, True], show_plot=False):
    """
    Make caterpillar plot of estimated parameter values.
    :param df: Melted dataframe for single group of metrics
    :param metric_dict: Dict of metric names/labels to use for melting/plotting
    :param xlabel: Label for x-axis
    :param output_path: Path to store output graph
    :param sort_list: List of booleans for ascending sort in strip plot
    :param show_plot: (Optional) Show output graph prior to storing
    :return:
    """
    # TODO: add error bars around estimates
    # TODO: jitter values by category
    meltedDf = pd.melt(df, id_vars='Country Name', value_vars=metric_dict.keys(), var_name='Metric', value_name='Value')
    meltedDf.replace({'Metric':metric_dict}, inplace=True)

    sns.set()
    ax = sns.stripplot(x='Value', y='Country Name',
                       data=meltedDf.sort_values(by=['Metric', 'Value'], ascending=sort_list), hue='Metric', size=9)
    fig = ax.get_figure()
    ax.set_ylabel('')
    ax.set_xlabel(xlabel, fontsize=16)
    fig.set_size_inches((10, 12))
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(fontsize=14, loc=7, frameon=True, facecolor='white')
    if show_plot:
        plt.show()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    # Clear figure
    fig.clf()

def makeBICPlot(df, output_path, show_plot=False):
    """
    Makes Figure 5, a sorted graph of the difference in BIC values between the Gompertz and logistic regressions
    :param df: Dataframe with delta_BIC data
    :param output_path: Path to store output graph
    :param show_plot: (Optional) Show output graph prior to storing
    :return:
    """
    sns.set()
    ax = sns.stripplot(x='delta_BIC', y='Country Name', data=df.sort_values(by='delta_BIC', ascending=False),
                       color='dodgerblue', size=9)
    fig = ax.get_figure()
    ax.set_ylabel('')
    ax.set_xlabel('$\Delta$BIC', fontsize=16)
    fig.set_size_inches((10, 12))
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=16)
    plt.axvline(x=0, ls='--')
    if show_plot:
        plt.show()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    # Clear figure
    fig.clf()
