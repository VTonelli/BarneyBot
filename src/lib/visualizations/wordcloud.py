from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(freqdict, cmap='viridis', title=None, plot=False):
    wordcloud = WordCloud(background_color = 'black', width = 800, height = 400,
                      colormap = cmap, max_words = 180, contour_width = 3,
                      max_font_size = 80, contour_color = 'steelblue',
                      random_state = 0)

    wordcloud.generate_from_frequencies(freqdict)
    if title:
        plt.title(title)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.figure()