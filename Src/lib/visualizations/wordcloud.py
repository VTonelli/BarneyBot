from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(freqdict, cmap='viridis', title=None):
    """
    Plots a WordCloud of a dictionary of frequencies of a given tv show character
    ## Params 
    * `freqdict`: dictionary of frequencies of a given tv show character
    * `cmap`: colormap for words in the wordcloud
    * `title`: title of plot
    """
    # initialize the wordcloud setting the plot parameters
    wordcloud = WordCloud(background_color = 'black', width = 800, height = 400,
                          colormap = cmap, max_words = 180, contour_width = 3,
                          max_font_size = 80, contour_color = 'steelblue',
                          random_state = 0)
    # generate the wordcloud from frequencies
    wordcloud.generate_from_frequencies(freqdict)
    # if title is not None
    if not (title is None):
        plt.title(title)
    # show the plot
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.figure()