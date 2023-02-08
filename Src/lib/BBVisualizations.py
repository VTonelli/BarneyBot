# reference to https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

from .visualizations.emotionsradar import EmotionsRadar
from .visualizations.wordcloud import plot_wordcloud
from .visualizations.metrics_plot import *
from . import BBData
from .BBData import EnumBase
from .BBMetrics import MetricsMTEnum, MetricsTGEnum, MetricsSSIMEnum, MetricsClsEnum
from .BBMetricResults import load_metric_by_name, MetricActor

class PlotsEnum(EnumBase):
    """
    Enumeration of all possible Plots
    * `MT` = "Machine Translation"
    * `TG` = "Text Generation"
    * `SS` = "Semantic Similarity"
    * `ECR` = "Emotion Classifier Radar"
    * `FCR` = "Frequency Classifier Radar"
    * `DBCR` = "DistilBert Classifier Radar"
    * `WC` = "Wordcloud"
    """
    MT = "Machine Translation"
    TG = "Text Generation"
    SS = "Semantic Similarity"
    ECR = "Emotion Classifier Radar"
    FCR = "Frequency Classifier Radar"
    DBCR = "DistilBert Classifier Radar"
    WC = "Wordcloud"


class BBVisualization:
    """
    Manage the visualization for every esperiment performed over the chatbot, according to
    a well defined list of tasks given by `PlotsEnum`.
    
    It is possible to load a BBVisualization object by the predefined static method `load_visualization`
    """
    # metric files location 
    METRIC_STORE_LOCATION_PATH = "../Metrics/New" 

    def __init__(self, name, visualization, metrics_data):
        self.name = name
        self.visualization = visualization
        self.metrics_data = metrics_data
        self.require_args = None
        self.optional_args = None
        
        if name == PlotsEnum.MT.value:
            self.require_args = set()
            self.optional_args = set(['logscale'])
        elif name == PlotsEnum.TG.value:
            self.require_args = set()
            self.optional_args = set(['logscale'])
        elif name == PlotsEnum.SS.value:
            self.require_args = set()
            self.optional_args = set(['logscale'])
        elif self.name == PlotsEnum.ECR.value or self.name == PlotsEnum.FCR.value:
            self.require_args = set([])
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
        Load a visualization ready to be plotted later calling the method `plot`, giving also different point of views of the evaluation.

        PLease notice that for each task there is a set of metrics allowed to be shown.
        ## Params
        * `name`: is the name of the visualization to load, namely the task tested we performed over the chatbots. This is
        a mandatory parameter
        * `characters`: list of characters to show in the plot, e.g. `['Barney', 'Sheldon']`
        * `metrics`: select the subset of metrics giving a list with name metrics which should be shown, e.g. if `name='Text Generation'`
        then metrics could be something like that `['BLEURT', 'Perplexity']`
        * `commondf`: asks to the visualizer to plot the test performed over the common dataset

        ## Return
        A `BBVisualization` initialize with the given parameters.
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
                # initialize the dictionary containing the test results foreach character
                mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
                # for each metric in the enumeration
                for m in MetricsMTEnum.tolist(): 
                    # s.t. m has been selected in metrics_list by the user
                    if m in metrics_list:
                        # load the results of the metric m
                        metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                        # search the right value to return given the actors
                        for c in characters:
                            for v in metric_dict_loaded.values():
                                F_is_actor = True
                                for actor in v['metric_actors'].values():
                                    F_is_actor = F_is_actor and (c in actor)
                            
                                if (F_is_actor): mt_dict[c].append(v['answer']['score'])
                # set the title
                title = PlotsEnum.MT.value + ' plot'
            else:
                # initialize the dictionary containing the test results foreach character
                mt_dict = {'metrics': metrics_list} | \
                          {commondf_key(c1, c2): [0 for _ in metrics_list] \
                            for c1 in characters for c2 in characters if c1 < c2}
                # for each metric in the enumeration
                for m in MetricsMTEnum.tolist(): 
                    # load the results of the metric m
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    # for eache value associated with the Common dataset search the right value to return
                    for v in [v for v in metric_dict_loaded.values() if v['reference_set'] == 'Common_df']:
                        actors = list(v['metric_actors'].values())
                        key = commondf_key(actors[0][1], actors[1][1])
                        m_value = mt_dict[key]
                        m_value[metrics_list.index(m)] = v['answer']['score']
                        mt_dict.update({key: m_value})
                # remove each value associated with a metric not selected 
                mt_dict1 = mt_dict.copy()
                for k1, v1 in zip(mt_dict.keys(), mt_dict.values()):
                    if v1 == [0 for _ in metrics_list]: mt_dict1.pop(k1, None)
                    mt_dict = mt_dict1
                # set the title for the plot
                title = PlotsEnum.MT.value + ' plot\n(over Common dataset)'
                
            if debug: print(mt_dict)
            # set the visualization
            visualization = BBVisualization(name, lambda: barplot(mt_dict, title), mt_dict)
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
                # initialize the dictionary containing the test results foreach character
                mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
                # for each metric in the enumeration s.t. m has been selected in metrics_list by the user
                for m in [m for m in MetricsTGEnum.tolist() if m in metrics_list]:
                    # load the results of the metric m
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    # search the right value to return given the actors
                    for c in characters:
                        for v in metric_dict_loaded.values():
                            F_is_actor = False
                            for actor in v['metric_actors'].values():
                                F_is_actor = F_is_actor or ([MetricActor.DIALOGPT_SAMPLE, c] == actor)
                        
                            if (F_is_actor) and (v['reference_set'] == c + '_df'): 
                                mt_dict[c].append(v['answer']['score'])    
                # set the title
                title = PlotsEnum.TG.value + ' plot'
            else:
                # initialize the dictionary containing the test results foreach character
                mt_dict = {'metrics': metrics_list} | \
                          {commondf_key(c1, c2): [0 for _ in metrics_list] \
                            for c1 in characters for c2 in characters if c1 < c2} | \
                          {c: [0 for _ in metrics_list] for c in characters}
                # for each metric in the enumeration
                for m in MetricsTGEnum.tolist(): 
                    # load the results of the metric m
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
                # remove each value associated with a metric not selected 
                mt_dict1 = mt_dict.copy()
                for k1, v1 in zip(mt_dict.keys(), mt_dict.values()):
                    if v1 == [0 for _ in metrics_list]: mt_dict1.pop(k1, None)
                    mt_dict = mt_dict1
                # set the title for the plot
                title = PlotsEnum.TG.value + ' plot\n(over Common dataset)'
            if debug: print(mt_dict)
            # set the visualization
            visualization = BBVisualization(name, 
                                            lambda l: barplot(mt_dict, title, logscale=l),
                                            mt_dict)
        ###
        elif name == PlotsEnum.SS.value:                # Semantic Similarity plot
            # Parameters preparation
            characters = kwargs['characters'] if 'characters' in kwargs else [c for c in BBData.character_dict][:-1]
            metrics_list = kwargs['metrics'] if 'metrics' in kwargs else MetricsSSIMEnum.tolist()
            debug = kwargs['debug'] if 'debug' in kwargs else False
            commondf = kwargs['commondf'] if 'commondf' in kwargs else False
            mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
            ##
            if not commondf:
                # initialize the dictionary containing the test results foreach character
                mt_dict = {'metrics': metrics_list} | {c: [] for c in characters}
                # for each metric in the enumeration s.t. m has been selected in metrics_list by the user
                for m in [m for m in MetricsSSIMEnum.tolist() if m in metrics_list]:
                    # load the results of the metric m
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    # search the right value to return given the actors
                    for c in characters:
                        for v in metric_dict_loaded.values():
                            F_is_actor = False
                            for actor in v['metric_actors'].values():
                                F_is_actor = F_is_actor or ([MetricActor.DIALOGPT_SAMPLE, c] == actor)
                        
                            if (F_is_actor) and (v['reference_set'] == c + '_df'): 
                                mt_dict[c].append(v['answer']['score'])    
                # set the title
                title = PlotsEnum.SS.value + ' plot'
            else:
                # initialize the dictionary containing the test results foreach character
                mt_dict = {'metrics': metrics_list} | \
                          {commondf_key(c1, c2): [0 for _ in metrics_list] \
                            for c1 in characters for c2 in characters if c1 < c2} | \
                          {c: [0 for _ in metrics_list] for c in characters}
                # for each metric in the enumeration
                for m in MetricsSSIMEnum.tolist(): 
                    # load the results of the metric m
                    metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, m)
                    # search the right value to return given the actors
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
                # set the title
                title = PlotsEnum.SS.value + ' plot\n(over Common dataset)'
            if debug: print(mt_dict)
            visualization = BBVisualization(name, 
                                            lambda l: barplot(mt_dict, title, logscale=l),
                                            mt_dict)
        ### 
        elif name == PlotsEnum.ECR.value:                # Emotion Radar
            # Parameters preparation
            if not 'character' in kwargs or type(kwargs['character']) != str: 
                raise Exception("One name of a character must be specified for visualize radarplot onclassification task")
            character = kwargs['character']
            debug = kwargs['debug'] if 'debug' in kwargs else False
            commondf = kwargs['commondf'] if 'commondf' in kwargs else False
            #
            # initialize the dictionary containing the test results foreach character
            metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, 
                                                     MetricsClsEnum.EMOTION_CLS.value)
            sources = None
            predictions = None
            labels = None
            # search the right value to return given the actors
            for v in metric_dict_loaded.values():
                actors = list(v['metric_actors'].values())[0]
                if actors == [MetricActor.DATASET_CHAR, character]:
                    sources = v['answer']['score']
                    if debug: print(sources)
                elif actors == [MetricActor.DIALOGPT_SAMPLE, character]:
                    predictions = v['answer']['score']
                    if debug: print(predictions)
                    labels = v['answer']['label']
                    if debug: print(labels)
            visualization = BBVisualization(name, 
                                            lambda : EmotionsRadar(labels, predictions, sources, character),
                                            None)
        ###
        elif name == PlotsEnum.FCR.value:                # Frequency Classifier Radar
            # Parameters preparation
            if not 'character' in kwargs or type(kwargs['character']) != str: 
                raise Exception("One name of a character must be specified for visualize radarplot onclassification task")
            character = kwargs['character']
            debug = kwargs['debug'] if 'debug' in kwargs else False
            #
            # initialize the dictionary containing the test results foreach character
            metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, 
                                                     MetricsClsEnum.FREQUENCY_CLS.value)
            sources = None
            predictions = None
            labels = None
            # search the right value to return given the actors
            for v in metric_dict_loaded.values():
                actors = list(v['metric_actors'].values())[0]
                if actors == [MetricActor.DATASET_CHAR, character]:
                    sources = v['answer']['score']
                    if debug: print(sources)
                elif actors == [MetricActor.DIALOGPT_SAMPLE, character]:
                    predictions = v['answer']['score']
                    if debug: print(predictions)
                    labels = v['answer']['label']
                    if debug: print(labels)
            visualization = BBVisualization(name, 
                                            lambda : EmotionsRadar(labels, predictions, sources, character),
                                            None)
        ###
        elif name == PlotsEnum.DBCR.value:               # DistilBert Classifier Radar
            # Parameters preparation
            if not 'character' in kwargs or type(kwargs['character']) != str: 
                raise Exception("One name of a character must be specified for visualize radarplot onclassification task")
            character = kwargs['character']
            debug = kwargs['debug'] if 'debug' in kwargs else False
            #
            # initialize the dictionary containing the test results foreach character
            metric_dict_loaded = load_metric_by_name(BBVisualization.METRIC_STORE_LOCATION_PATH, 
                                                     MetricsClsEnum.DISTILBERT_CLS.value)
            sources = None
            predictions = None
            labels = None
            # search the right value to return given the actors
            for v in metric_dict_loaded.values():
                actors = list(v['metric_actors'].values())[0]
                if actors == [MetricActor.DATASET_CHAR, character]:
                    sources = v['answer']['score']
                    if debug: print(sources)
                elif actors == [MetricActor.DIALOGPT_SAMPLE, character]:
                    predictions = v['answer']['score']
                    if debug: print(predictions)
                    labels = v['answer']['label']
                    if debug: print(labels)
            visualization = BBVisualization(name, 
                                            lambda : EmotionsRadar(labels, predictions, sources, character),
                                            None)
        ###
        elif name == "wordcloud":
            visualization = BBVisualization(name, lambda f: plot_wordcloud(f), None)
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
        elif self.name == PlotsEnum.SS.value:
            self.visualization(kwargs['logscale'] if 'logscale' in kwargs else False)
        elif self.name == PlotsEnum.ECR.value:
            radar = self.visualization()
            radar.plotEmotionsRadar(PlotsEnum.ECR.value)
        elif self.name == PlotsEnum.FCR.value:
            radar = self.visualization()
            radar.plotEmotionsRadar(PlotsEnum.FCR.value)
        # elif self.name == "wordcloud":
        #     self.visualization(kwargs['freqdict'])

    def corr(self, correlate='characters', debug=False):
        '''
        Allows to plot a correlation matrix over the task tested which had been previously loaded
        # Params
        * `correlate` = 'characters' | 'metrics'
        '''
        if self.metrics_data is None:
            raise("A metric set must be loaded before to run the correlation!")
        if not (correlate in ['characters','metrics']):
            raise("You can correlate only characters or metrics!")
        corrplot(self.metrics_data, correlate=='metrics', self.name, debug)
        