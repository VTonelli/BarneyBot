# reference to https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

from .visualizations.emotionsradar import EmotionsRadar
from .visualizations.wordcloud import plot_wordcloud

class BBVisualization:
    visualizations_list = [
        "emotions radar"
    ]
    
    def __init__(self, name, visualization):
        self.visualization = visualization
        self.name = name
        self.require_args = None
        self.optional_args = None
        
        if name == "emotions radar":
            self.require_args = set(["emotions", "labels", "predictions", "character"])
            self.optional_args = set()
        elif name == "wordcloud":
            self.require_args = set(['freqdict'])
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
        if name == "emotions radar":
            visualization = BBVisualization(name, lambda e, p, l, c: EmotionsRadar(e, p, l, c))
        elif name == "wordcloud":
            visualization = BBVisualization(name, lambda f: plot_wordcloud(f))
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
        if self.name == "emotions radar":
            radar = self.visualization(kwargs['emotions'], kwargs['predictions'], kwargs['labels'], kwargs['character'])
            radar.plotEmotionsRadar()
        elif self.name == "wordcloud":
            self.visualization(kwargs['freqdict'])