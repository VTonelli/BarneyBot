from matplotlib import pyplot as plt
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
    ax.set_ylim(bottom=-2)
    ax.grid()
    plt.show()
