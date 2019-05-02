import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib.patches as mpatches

CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from datasets import Data, get_data_folder
from fns_categorical_encoding import file_meet_conditions2, read_json
from fns_figures_dataset import set_list

plt.close('all')
# PARAMETERS ##################################################################
sns.set_style("ticks")

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

dataset_cm = {'docs_payments': 0,
              'midwest_survey': 1,
              'medical_charge': 2,
              'beer_reviews': 3,
              'beer_reviews2': 4,
              'road_safety': 5,
              'employee_salaries': 6,
              'indultos_espana': 7}
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
# 'indultos_espana', 'sydney_cycling'
linewidth = 1.5
results_path = os.path.join(get_data_folder(), 'results', 'ecml2018')
path = os.path.join(CE_HOME, 'submissions', 'ecml2018')
##############################################################################

datasets_name = {'employee_salaries': 'employee\nsalaries',
                 'traffic_violations': 'traffic\nviolations',
                 'beer_reviews': 'beer\nreviews',
                 'beer_reviews2': 'beer\nreviews 2',
                 'midwest_survey': 'midwest\nsurvey',
                 'docs_payments': 'open\npayments',
                 'medical_charge': 'medical\ncharges',
                 'road_safety': 'road\nsafety'
                 }

plt.close('all')
df_all = pd.DataFrame()
score_type = {}
dimSE_nored = {}  # change this for a friendlier method
for dataset in datasets:
    data = Data(dataset)
    if dataset in ['docs_payments', 'crime_data', 'beer_reviews2',
                   'traffic_violations']:
        n_rows = 100000  # -1 if using all rows for prediction
    elif dataset in ['beer_reviews', 'road_safety']:
        n_rows = 10000
    else:
        n_rows = -1
    if dataset in ['adult', 'adult2', 'adult3']:
        typo_prob = .1
    else:
        typo_prob = 0
    conditions = {'dataset': data.name,
                  'n_splits': 100,
                  'test_size': .2,
                  'n_rows': n_rows,
                  'str_preprocess': True,
                  'encoder': ['3gram_SimilarityEncoder',
                              'one-hot_encoding_sparse'],
                  'clf': ['LogisticRegression', 'RandomForest',
                          'GradientBoosting', 'MLP', 'Ridge'],
                  #   'col_action': data.col_action,
                  #   'clf_type': data.clf_type
                  }
    print(dataset)
    files = glob(os.path.join(results_path, '*.json'))
    files2 = file_meet_conditions2(files, conditions)
    if len(files2) == 0:
        continue
    print('Relevant files: %d' % len(files2))
    min_d = np.infty  # change this for a friendlier method
    for f in files2:
        f_dict = read_json(f)
        df = pd.DataFrame(f_dict['results'])
        # df = df.drop_duplicates(subset=df.columns[1:])
        df['dataset'] = data.name
        df['encoder'] = f_dict['encoder']
        # clf type
        if f_dict['clf_type'] == 'regression':
            score_type[dataset] = 'r2 score'
        if f_dict['clf_type'] == 'binary-clf':
            score_type[dataset] = 'average precision score'
        if f_dict['clf_type'] == 'multiclass-clf':
            score_type[dataset] = 'accuracy score'
        df['score_type'] = score_type[dataset]
        df['clf'] = [clf for clf in set_list('Classifier', f_dict['clf_type'])
                     if clf in f_dict['clf'][0]][0]
        dim_red_method = (('%s (d=%d) %s')
                          % (f_dict['dimension_reduction'][0],
                          f_dict['dimension_reduction'][1],
                          [clf for clf in set_list('Classifier')
                           if clf in f_dict['clf'][0]][0]))
        min_d = min(f_dict['dimension_reduction'][1], min_d)
        df['Dimension reduction method'] = dim_red_method
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
    max_f = df_all[(df_all['fold'] == 1) &
                   (df_all['dataset'] == dataset)
                   ]['n_train_features'].max()
    min_f = df_all[(df_all['fold'] == 1) &
                   (df_all['dataset'] == dataset)
                   ]['n_train_features'].min()
    dimSE_nored[dataset] = min_d+max_f-min_f
df_all[df_all['dataset'] == 'docs_payments']
df_all['encoder']
# Figure 3a
methods = sorted(df_all['Dimension reduction method'].unique())
methods = ['- (d=-1) LogisticRegression',
           '- (d=-1) Ridge',
           '- (d=-1) GradientBoosting',
           '- (d=-1) RandomForest',
           ][::-1]
methods_dict = {'- (d=-1) MLP': 'MLP',
                '- (d=-1) LogisticRegression': 'Logistic CV',
                '- (d=-1) Ridge': 'Ridge CV',
                '- (d=-1) RandomForest': 'Random Forest',
                '- (d=-1) GradientBoosting': 'Gradient Boosting'}
palette_sns = sns.color_palette("hls", len(methods)+1)
palette = [palette_sns[4],
           palette_sns[1],
           palette_sns[2],
           palette_sns[3]][::-1]
palette_dict = {method: palette[i] for i, method in enumerate(methods)}
fontsize = 26
clfs = sorted(df_all['clf'].unique())
classifiers = sorted(df_all['clf'].unique())
score = 'score'

# iterate over methods for each dataset
fig, ax = plt.subplots(1, len(datasets),
                       figsize=(len(datasets)*2, len(methods)))
plt.subplots_adjust(wspace=.1)
methods_by_dataset = {dataset: 0 for dataset in datasets}
medians_dict = {}
rank_medians_dict = {}
for i, dataset in enumerate(datasets):
    X = []
    Y = []
    av_train_time = []
    labels = []
    for method in methods:
        classifier = method.split(' ')[-1]
        df_y = df_all[(df_all['dataset'] == dataset) &
                      (df_all['clf'] == classifier) &
                      (df_all['Dimension reduction method'] == method) &
                      (df_all['encoder'] == 'one-hot_encoding_sparse')]
        # if (classifier == 'LogisticRegression') & (df.shape[0] == 0):
        #     df = df_all[(df_all['dataset'] == dataset) &
        #                 (df_all['clf'] == 'Ridge') &
        #                 (df_all['Dimension reduction method'] ==
        #                  method.split('LogisticRegression')[0] + 'Ridge')]
        y = df_y[score].values
        if len(y) > 100:
            if sum(y[0:100] == y[100:]) == 100:
                y = y[0:100]
            else:
                # raise ValueError('Something weird happens')
                y = y[0:100]
                print('Something weird happens')
        df = df_all[(df_all['dataset'] == dataset) &
                    (df_all['clf'] == classifier) &
                    (df_all['Dimension reduction method'] == method) &
                    (df_all['encoder'] == '3gram_SimilarityEncoder')]
        # if (classifier == 'LogisticRegression') & (df.shape[0] == 0):
        #     df = df_all[(df_all['dataset'] == dataset) &
        #                 (df_all['clf'] == 'Ridge') &
        #                 (df_all['Dimension reduction method'] ==
        #                  method.split('LogisticRegression')[0] + 'Ridge')]
        av_train_time.append(df['training_time'].mean())
        x = df[score].values
        labels.append(method)
        if len(x) > 100:
            if sum(x[0:100] == x[100:]) == 100:
                x = x[0:100]
            else:
                # raise ValueError('Something weird happens')
                x = x[0:100]
                print('Something weird happens')
        try:
            if len(x) == 0:
                X.append(x)
            else:
                X.append(x)
        except ValueError:
            X.append([])
        try:
            if len(y) == 0:
                Y.append(y)
            else:
                Y.append(y)
        except ValueError:
            Y.append([])
        methods_by_dataset[dataset] += 1
    medianprops = dict(linestyle='-', linewidth=linewidth,
                       color='red')
    n1 = len(methods)
    n2 = len(datasets)
    positions = np.linspace(1, len(methods), len(methods))
    bplot2 = ax[i].boxplot(Y, positions=positions+.23,
                           labels=labels, vert=False, patch_artist=True,
                           medianprops=medianprops, widths=.4, zorder=4,
                           showfliers=False)
    bplot = ax[i].boxplot(X, positions=positions-.23,
                          labels=labels, vert=False, patch_artist=True,
                          medianprops=medianprops, widths=.4, zorder=3,
                          showfliers=False)
    for box, label in zip(bplot2['boxes'], labels):
        box.set(linewidth=linewidth, color='darkslateblue')
        box.set(facecolor='#bababa')
    for cap in bplot2['caps']:
        cap.set(linewidth=linewidth, color='darkslateblue')
    for whisker in bplot2['whiskers']:
        whisker.set(linewidth=linewidth, color='darkslateblue')
    for box, label in zip(bplot['boxes'], labels):
        box.set(linewidth=linewidth, color='darkslateblue')
        box.set(facecolor=sns.color_palette('pastel')[2])
    for cap in bplot['caps']:
        cap.set(linewidth=linewidth, color='darkslateblue')
    for whisker in bplot['whiskers']:
        whisker.set(linewidth=linewidth, color='darkslateblue')
    ymin, ymax = ax[i].get_ylim()
    ymin, ymax = ax[i].set_ylim([0.5, len(methods) + .5])
    xmin, xmax = ax[i].get_xlim()
    # tick rotation
    # for label in ax[i].get_xmajorticklabels():
    #     label.set_rotation(30)
    #     label.set_horizontalalignment("right")
    ax[i].tick_params(axis='y', which='major', labelsize=fontsize)
    ax[i].set_xlabel(datasets_name[dataset], labelpad=10,
                     fontsize=fontsize, rotation=0)
    if i == 0:
        methods_vals = [methods_dict[method] for method in methods]
        ax[0].set_yticks([1, 2, 3, 4])
        ax[0].set_yticklabels(methods_vals, ha='right')
        # align yticklabels to the left
        # plt.draw()  # because get_window_extent needs a renderer to work
        # yax = ax[i].get_yaxis()
        # pad = max(T.label.get_window_extent().width
        #           for T in yax.majorTicks)+20
        # yax.set_tick_params(pad=pad)
    else:
        ax[i].set_yticklabels([])
        ax[i].tick_params(axis='y', which='both', length=0)
    # putting an asterisc on the prediction with the maximum median
    medians = np.array([np.median(x) if len(x) > 0 else np.nan for x in X])
    medians_ohe = np.array([np.median(y) if len(y) > 0 else np.nan for y in Y])
    medians_dict[dataset] = medians
    max_median_method = np.argsort(medians)[::-1]
    rank_medians_dict[dataset] = ss.rankdata(-medians)
    rank_medians_dict[dataset][np.isnan(medians)] = np.nan
    # ax[i].text(xmin+(xmax-xmin)*.025, max_median_method[0]+1.4,
    #            '1st', fontsize=fontsize, ha='left', va='top',
    #            color='black')
    y_i = ymin
    delta = (ymax-ymin)/len(methods)
    for j, method in enumerate(methods):
        y_f = y_i + delta
        if j >= 0:
            if j/2.0 != int(j/2):
                ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#ffffff')
            else:
                ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#f4f4f4')
        else:
            if j/2.0 != int(j/2):
                ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#fff2f2')
            else:
                ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#ffe8e8')
        if i == len(datasets)-1:
            xmin_, xmax_ = 1.05, 1.37
            if j >= 0:
                if j/2.0 != int(j/2):
                    ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                  color='#ffffff', clip_on=False, zorder=2)
                else:
                    ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                  color='#f4f4f4', clip_on=False, zorder=2)
            else:
                if j/2.0 != int(j/2):
                    ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                  color='#fff2f2', clip_on=False, zorder=2)
                else:
                    ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                  color='#ffe8e8', clip_on=False, zorder=2)
        ax[i].vlines(x=medians_ohe[j], ymin=y_i, ymax=y_f,
                     linewidth=linewidth, color='grey',
                     linestyle='--', zorder=2)
        y_i = y_f
# average ranking
for i in [len(datasets)-1]:
    ymin, ymax = ax[i].get_ylim()
    xmin, xmax = ax[i].get_xlim()
    average_ranking = [np.nanmean([rank_medians_dict[ds][i]
                                   for ds in datasets])
                       for i in range(len(methods))]
    for m in range(len(methods)):
        ax[i].text(xmax+(xmax-xmin)*.34, m+1.0,
                   '%.1f' % average_ranking[m], fontsize=fontsize-3,
                   ha='right', va='center', color='black',
                   weight='normal')
    ax[i].axhspan(ymin, ymax, xmin=xmin_, xmax=xmax_, facecolor='None',
                  edgecolor='black', clip_on=False, linewidth=linewidth*.7,
                  zorder=4)
    ax[i].text(xmax+(xmax-xmin)*.62, 1.1*ymax,
               'Avgerage ranking across datasets', fontsize=fontsize-2,
               ha='right', va='top', color='black', rotation=-90,
               weight='normal')

# xticks
for i in range(len(datasets)):
    plt.draw()
    ax[i].tick_params(axis='x', which='major', labelsize=fontsize-5)
    if len(ax[i].get_xticklabels()) > 4:
        ax[i].set_xticks(ax[i].get_xticks()[np.array([0, 1, 3, 4])])
        ax[i].set_xticklabels([l.get_text() for j, l
                               in enumerate(ax[i].get_xticklabels())
                               if j in [0, 1, 3, 4]])
patches = []
patch = mpatches.Patch(color='#bababa', label='one-hot encoding',
                       linewidth=linewidth)
patch.set_edgecolor('darkslateblue')
patches.append(patch)
patch = mpatches.Patch(color=sns.color_palette('pastel')[2],
                       label='3-gram similarity encoding', linewidth=linewidth)
patch.set_edgecolor('darkslateblue')
patches.append(patch)
leg = plt.legend(handles=patches,
                 loc='upper left',
                 bbox_to_anchor=(-6, 1.28),
                 ncol=2,
                 fontsize=fontsize-2,
                 frameon=True,
                 edgecolor='None')
leg.set_zorder(2)
# plt.suptitle('Scores by dataset with different classifiers. ' +
#              'Similarity measure: %s' % '3-gram',
#              fontsize=fontsize, weight='bold', y=1, x=+.1, ha='left')
figname = ('datasets_3gram_classifiers_scorediff.pdf')
print('Saving: ' + figname)
plt.savefig(os.path.join(path, figname),
            transparent=False, bbox_inches='tight', pad_inches=0.2)
