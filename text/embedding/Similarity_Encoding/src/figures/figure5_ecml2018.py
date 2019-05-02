import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import jellyfish
import Levenshtein as lev
import ngram

CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from datasets import Data, get_data_folder
from fns_categorical_encoding import file_meet_conditions2, read_json
from fns_figures_dataset import set_list, random_combination

# PARAMETERS ##################################################################
path = os.path.join(CE_HOME, 'submissions', 'ecml2018')

markers = {'o': 'circle',
           'v': 'triangle_down',
           '^': 'triangle_up',
           's': 'square',
           'D': 'diamond',
           'P': 'plus_filled',
           'X': 'x_filled',
           '*': 'star'}
datasets = ['medical_charge',
            'employee_salaries',
            'docs_payments',
            'midwest_survey',
            'traffic_violations',
            'road_safety',
            'beer_reviews'
            ]
distances = ['jaro-winkler_SimilarityEncoder',
             'levenshtein-ratio_SimilarityEncoder',
             '3gram_SimilarityEncoder']

##############################################################################
# Figure 1: histogram: number of elements in the ball for each sample

palette = sns.color_palette('pastel')
color = {'jaro-winkler_SimilarityEncoder': 0,
         'levenshtein-ratio_SimilarityEncoder': 1,
         '3gram_SimilarityEncoder': 2}
dist_dict = {'jaro-winkler_SimilarityEncoder': 'Jaro-winkler',
             'levenshtein-ratio_SimilarityEncoder': 'Levenshtein-ratio',
             '3gram_SimilarityEncoder': '3-gram'}
datasets_name = {'employee_salaries': 'employee\nsalaries',
                 'traffic_violations': 'traffic\nviolations',
                 'beer_reviews': 'beer\nreviews',
                 'midwest_survey': 'midwest\nsurvey',
                 'docs_payments': 'open\npayments',
                 'medical_charge': 'medical\ncharges',
                 'road_safety': 'road\nsafety'
                 }
X = {dataset: dict() for dataset in datasets}
median = {dataset: dict() for dataset in datasets}
for dataset in datasets:
    print(dataset)
    data = Data(dataset).get_df(preprocess_df=True)
    if dataset in ['docs_payments', 'crime_data', 'beer_reviews2',
                   'traffic_violations']:
        n_rows = 100000  # -1 if using all rows for prediction
    elif dataset in ['beer_reviews', 'road_safety']:
        n_rows = 10000
    else:
        n_rows = -1
    df = data.df.sample(frac=1, random_state=5).reset_index(drop=True)[:n_rows]

    SE_var = [col for col in data.col_action
              if data.col_action[col] == 'se'][0]
    SE_cats = df[SE_var].unique()
    m = len(SE_cats)

    n = 10000
    pairs = random_combination(df[SE_var], n, 2)
    for distance in distances:
        values = []
        for pair in pairs:
            if distance == '3gram_SimilarityEncoder':
                values.append(ngram.NGram.compare(str(pair[0]), str(pair[1])))
            if distance == 'levenshtein-ratio_SimilarityEncoder':
                values.append(lev.ratio(str(pair[0]), str(pair[1])))
            if distance == 'jaro-winkler_SimilarityEncoder':
                values.append(
                    jellyfish.jaro_winkler(str(pair[0]), str(pair[1])))

        label = dataset
        median[dataset][distance] = np.median(values)
        x = np.histogram(values, bins=np.linspace(0, .999, 20))
        x = (np.log10(x[0]), x[1])
        X[dataset][distance] = x

palette = sns.color_palette('pastel')
sns.set_style("ticks")
fig, ax = plt.subplots(len(distances), sharex=True,
                       figsize=(len(datasets)*1.8/1.5, len(distances)/1.5))
plt.subplots_adjust(hspace=.02)
fontsize = 14
linewidth = 1

plt.rc('text', usetex=True)
params = {
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'mathtext.bf': 'serif:bold',
    'mathtext.it': 'serif:italic',
    'mathtext.sf': 'sans\\-serif',
    'font.family': 'serif',
    'font.serif': "Computer Modern",  # or "Times"
    'text.latex.preamble': [
         r'\usepackage{siunitx}',
         r'\usepackage{amsmath}',
         r'\usepackage{libertineRoman}',
         r'\usepackage[libertine]{newtxmath}'
         ]
          }
plt.rcParams.update(params)

for i, dist in enumerate(distances):
    for j, ds in enumerate(datasets):
        delta = X[ds][dist][1][1]/2
        ax[i].fill_between(X[ds][dist][1] + 1.2*j,
                           np.maximum(0, (list(X[ds][dist][0]) +
                                          [X[ds][dist][0][-1]])),
                           y2=0,
                           label=dist_dict[dist],
                           color=palette[color[dist]])
        ax[i].plot(X[ds][dist][1] + 1.2*j,
                   np.maximum(0, (list(X[ds][dist][0]) +
                                  [X[ds][dist][0][-1]])),
                   linewidth=linewidth,
                   color='darkslateblue')
        ax[i].vlines(x=median[ds][dist] + 1.2*j, ymin=0, ymax=4,
                     linewidth=linewidth+.5, color='red',
                     linestyle='-', zorder=2)
        ax[i].set_axis_off()
    ymin, ymax = ax[i].get_ylim()
    ax[i].text(-.1, (ymax-ymin)/5,
               dist_dict[dist],
               fontsize=fontsize,
               horizontalalignment='right', verticalalignment='center',
               color='black')
    ax[i].set_yticks([])
for j, ds in enumerate(datasets):
    ax[i].text(.5 + 1.2*j, -5.5,
               datasets_name[ds],
               fontsize=fontsize,
               horizontalalignment='center', verticalalignment='center',
               color='black')
xticks = np.concatenate([np.array([0, .5, 1]) + 1.2*j
                         for j in range(len(datasets))])
xticklabels = np.concatenate([np.array(['0', '0.5', '1'])
                              for j in range(len(datasets))])
for j in range(len(xticks)):
    ax[i].text(xticks[j], -2,
               xticklabels[j],
               fontsize=fontsize-2,
               horizontalalignment='center', verticalalignment='center',
               color='gray')
for i in range(len(distances)):
    ymin, ymax = ax[i].get_ylim()
    for j, ds in enumerate(datasets):
        ax[i].hlines(ymin, -.007 + 1.2*j, 1.007+1.2*j, linewidth=linewidth+.4,
                     color='gray')
    ax[i].vlines(xticks, ymin, ymin-.7, linewidth=linewidth, color='gray')

patches = []
for distance in distances:
    patch = mpatches.Patch(color=palette[color[distance]],
                           label=dist_dict[distance], linewidth=linewidth)
    patch.set_edgecolor('darkslateblue')
    patches.append(patch)

# plt.show()

figname = ('histogram_distances.pdf')
print('Saving: ' + figname)
plt.savefig(os.path.join(path, figname),
            transparent=False, bbox_inches='tight', pad_inches=0.2)
