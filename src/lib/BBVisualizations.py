# reference to https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

from .visualizations.emotionsradar import EmotionsRadar
from .visualizations.wordcloud import plot_wordcloud
from .visualizations.datasetsbar import plot_datasets

from .BBData import EnumBase 
from .BBMetrics import MetricsEnum 

class PlotsEnum(EnumBase):
    """Other possible plots we can use to plot other relevant stuff for the project"""
    DATASETS = 'datasets'
    WORDCLOUD = 'wordcloud'

class BBVisualization:

    visualizations_list = MetricsEnum.tolist() + PlotsEnum.tolist()

    def __init__(self, name, visualization):
        self.visualization = visualization
        self.name = name
        self.require_args = None
        self.optional_args = None
        
        if name == PlotsEnum.DATASETS:
            self.require_args = set(["characters_datasets"])
            self.optional_args = set(['colors', 'out_plot_folder'])
        elif name == PlotsEnum.WORDCLOUD:
            self.require_args = set(['freqdict'])
            self.optional_args = set()
        elif name == MetricsEnum.EMOTION_CLS:
            self.require_args = set(["emotions", "labels", "predictions", "character"])
            self.optional_args = set()
   
    def __str__(self):
        return str({
            "instance": self,
            "name": self.name,
            "visualization": self.visualization
        })
    
    @staticmethod
    def load_visualization(name, **kwargs):
        visualization = None
        if name == PlotsEnum.DATASETS:
            visualization = BBVisualization(name, lambda d, c, g: plot_datasets(d, c, g))
        elif name == PlotsEnum.WORDCLOUD:
            visualization = BBVisualization(name, lambda f: plot_wordcloud(f))
        elif name == "emotions radar":
            visualization = BBVisualization(name, lambda e, p, l, c: EmotionsRadar(e, p, l, c))
        else:
            raise Exception("Unknown visualization name!")
        return visualization
    
    def plot(self, **kwargs):
        if not set(kwargs.keys()).issubset(
                set(self.require_args).union(
                    set(self.optional_args))):
            raise Exception("Unexpected arguments! Required arguments are",
                            self.require_args)
        if not set(self.require_args).issubset(set(kwargs.keys())):
            raise Exception("Missing arguments! Required arguments are",
                            self.require_args)
        if self.name == PlotsEnum.DATASETS:
            if not 'colors' in kwargs: 
                kwargs['colors'] = ['red','orange','gold']
            if not 'out_plot_folder' in kwargs:
                kwargs['out_plot_folder'] = None
            self.visualization(kwargs['characters_datasets'], kwargs['colors'], kwargs['out_plot_folder']) 
        elif self.name == PlotsEnum.WORDCLOUD:
            self.visualization(kwargs['freqdict'])
        elif self.name == MetricsEnum.EMOTION_CLS:
            radar = self.visualization(kwargs['emotions'], kwargs['predictions'], kwargs['labels'], kwargs['character'])
            radar.plotEmotionsRadar()
        