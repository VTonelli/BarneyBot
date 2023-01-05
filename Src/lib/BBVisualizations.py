# reference to https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

from .visualizations.emotionsradar import EmotionsRadar
from .visualizations.wordcloud import plot_wordcloud
from .visualizations.metrics_plot import *
from . import BBData
from .BBData import EnumBase
from .BBMetrics import MetricsMTEnum, MetricsTGEnum, MetricsSSIMEnum, MetricsClsEnum
from .BBMetricResults import load_metric_by_name, MetricActor

class PlotsEnum(EnumBase):
    """Plots"""
    MT = "Machine Translation"
    TG = "Text Generation"

class BBVisualization:
    visualizations_list = [
        "machine translation",
        "text generation", 
        "emotions radar", 
        "wordcloud"
    ]

    # metric files location 
    METRIC_STORE_LOCATION_PATH = "../Metrics/New" 

    def __init__(self, name, visualization):
        self.visualization = visualization
        self.name = name
        self.require_args = None
        self.optional_args = None
        
        if name == PlotsEnum.MT.value:
            self.require_args = set()
            self.optional_args = set(['metrics'])
        elif name == PlotsEnum.TG.value:
            self.require_args = set()
            self.optional_args = set(['logscale', 'metrics'])
        elif name == "emotions radar":
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
        """
        Load a visualization ready to be plotted later calling the method `plot`.
        """
        commondf_key = lambda c1, c2: c1+' vs '+c2 if c1 < c2 else c2+' vs '+c1

        visualization = None
        ###
        if name == PlotsEnum.MT.value:                  # Machine Translation plot
            # Parameters preparation
            characters = kwargs['characters'] if 'characters' in kwargs else [c for c in BBData.character_dict][:-1]
            metrics_list = kwargs['metrics'] if 'metrics' in kwargs else MetricsMTEnum.tolist()
            debug = kwargs['debug'] if 'debug' in kwargs else False
            commondf = kwargs['commondf'] if 'commondf' in kwargs else False
            ##
            if not commondf:
                mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
                for m in MetricsMTEnum.tolist(): 
                    if m in metrics_list:
                        metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                        if not commondf:
                            for c in characters:
                                for v in metric_dict_loaded.values():
                                    F_is_actor = True
                                    for actor in v['metric_actors'].values():
                                        F_is_actor = F_is_actor and (c in actor)
                                
                                    if (F_is_actor): mt_dict[c].append(v['answer']['score'])
                title = PlotsEnum.MT.value + ' plot'
            else:
                mt_dict = {'metrics': metrics_list} | \
                          {commondf_key(c1, c2): [0 for _ in metrics_list] \
                            for c1 in characters for c2 in characters if c1 < c2}
                for m in MetricsMTEnum.tolist(): 
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    for v in [v for v in metric_dict_loaded.values() if v['reference_set'] == 'Common_df']:
                        actors = list(v['metric_actors'].values())
                        key = commondf_key(actors[0][1], actors[1][1])
                        m_value = mt_dict[key]
                        m_value[metrics_list.index(m)] = v['answer']['score']
                        mt_dict.update({key: m_value})
                mt_dict1 = mt_dict.copy()
                for k1, v1 in zip(mt_dict.keys(), mt_dict.values()):
                    if v1 == [0 for _ in metrics_list]: mt_dict1.pop(k1, None)
                    mt_dict = mt_dict1
                title = PlotsEnum.MT.value + ' plot\n(over Common dataset)'
                
            if debug: print(mt_dict)
            visualization = BBVisualization(name, lambda: barplot(mt_dict, title))
        ###
        elif name == PlotsEnum.TG.value:                # Text Generation plot
            # Parameters preparation
            characters = kwargs['characters'] if 'characters' in kwargs else [c for c in BBData.character_dict][:-1]
            metrics_list = kwargs['metrics'] if 'metrics' in kwargs else MetricsTGEnum.tolist()
            debug = kwargs['debug'] if 'debug' in kwargs else False
            commondf = kwargs['commondf'] if 'commondf' in kwargs else False
            mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
            ##
            if not commondf:
                mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
                for m in [m for m in MetricsTGEnum.tolist() if m in metrics_list]:
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    for c in characters:
                        for v in metric_dict_loaded.values():
                            F_is_actor = False
                            for actor in v['metric_actors'].values():
                                F_is_actor = F_is_actor or ([MetricActor.DIALOGPT_SAMPLE, c] == actor)
                        
                            if (F_is_actor) and (v['reference_set'] == c + '_df'): 
                                mt_dict[c].append(v['answer']['score'])    
                title = PlotsEnum.TG.value + ' plot'
            else:
                mt_dict = {'metrics': metrics_list} | \
                          {commondf_key(c1, c2): [0 for _ in metrics_list] \
                            for c1 in characters for c2 in characters if c1 < c2} | \
                          {c: [0 for _ in metrics_list] for c in characters}
                for m in MetricsTGEnum.tolist(): 
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    for v in [v for v in metric_dict_loaded.values() if v['reference_set'] == 'Common_df']:
                        actors = list(v['metric_actors'].values())
                        if len(actors) == 1:
                            key = actors[0][1]
                        elif len(actors) == 2:
                            key = commondf_key(actors[0][1], actors[1][1])
                        else:
                            raise Exception('This metric has 3 or more actors! Metric name: '+m)
                        try:
                            m_value = mt_dict[key]
                            m_value[metrics_list.index(m)] = v['answer']['score']
                            mt_dict.update({key: m_value})
                        except:
                            continue
                    title = PlotsEnum.TG.value + ' plot\n(over Common dataset)'
                mt_dict1 = mt_dict.copy()
                for k1, v1 in zip(mt_dict.keys(), mt_dict.values()):
                    if v1 == [0 for _ in metrics_list]: mt_dict1.pop(k1, None)
                    mt_dict = mt_dict1
            if debug: print(mt_dict)
            visualization = BBVisualization(name, lambda l: barplot(mt_dict, title, 
                                                                    logscale=l))
        ###
        elif name == "emotions radar":
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
        print(self.name)

        if self.name == PlotsEnum.MT.value:
            self.visualization()
        elif self.name == PlotsEnum.TG.value:
            self.visualization(kwargs['logscale'] if 'logscale' in kwargs else False)
        # elif self.name == "emotions radar":
        #     radar = self.visualization(kwargs['emotions'], kwargs['predictions'], kwargs['labels'], kwargs['character'])
        #     radar.plotEmotionsRadar()
        # elif self.name == "wordcloud":
        #     self.visualization(kwargs['freqdict'])