from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def barplot(metrics: dict, title: str,
            logscale=True, figsize=(16,12), colormap='viridis'):
    df = pd.DataFrame(data=metrics)
    ax = df.plot(kind='bar', x='metrics',  
                 colormap=colormap, figsize=figsize)
    if logscale:
        ax.set_yscale('symlog')
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel('score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid()
    plt.show()

def corrplot(metrics: dict, inverted: bool, title: str, debug=False):
    if debug: print(metrics)
    df = pd.DataFrame(data=metrics)
    if debug: print(df)
    if inverted:
        dfT = df.T
        #set column names equal to values in row index position 0
        dfT.columns = dfT.loc['metrics']
        #remove first row from DataFrame
        dfT = dfT.drop('metrics')
        #reset index
        dfT.reset_index(inplace=True, names=['characters']+df['metrics'].to_list())
        #change dtypes
        for c in dfT.columns[1:]:
            dfT[c] = pd.to_numeric(dfT[c])
        dfT = dfT.drop('characters', axis=1)
        df = dfT
    else:
        df = df.drop('metrics', axis=1)
    if debug: print(df)
    corr = df.corr()
    # plt.pcolor(corr)
    sns.heatmap(corr, annot = True)
    plt.title(title)
    plt.show()

def corrm(corr: any, title: str):
    """plots a correlation matrix with its title"""
    plt.figure(figsize=(18,16))
    sns.heatmap(corr, annot = True)
    plt.title(title)
    plt.show()