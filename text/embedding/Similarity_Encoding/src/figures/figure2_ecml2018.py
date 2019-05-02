import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from datasets import Data, get_data_folder
from fns_categorical_encoding import file_meet_conditions2, read_json
from fns_figures_dataset import set_list
# import matplotlib.patches as mpatches
import scipy.stats as ss

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
    'font.family': 'serif:bold',
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
datasets = [
            'medical_charge',
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

clfs = ['Ridge', 'GradientBoosting']
for clf in clfs:
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
                      'encoder': [
                          'TargetEncoder',
                          'one-hot_encoding_sparse',
                          '3gram_SimilarityEncoder',
                          'jaro-winkler_SimilarityEncoder',
                          'levenshtein-ratio_SimilarityEncoder',
                          '3grams_count_vectorizer',
                          'HashingEncoder',
                          'MDVEncoder'],
                      'clf': [clf]
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
            df['clf'] = f_dict['clf_type'][0]
            df['encoder'] = f_dict['encoder']
            # clf type
            if f_dict['clf_type'] == 'regression':
                score_type[dataset] = 'r2 score'
            if f_dict['clf_type'] == 'binary-clf':
                score_type[dataset] = 'average precision score'
            if f_dict['clf_type'] == 'multiclass-clf':
                score_type[dataset] = 'accuracy score'
            df['score_type'] = score_type[dataset]
            dim_red_method = (('%s (d=%d) %s')
                              % (f_dict['dimension_reduction'][0],
                              f_dict['dimension_reduction'][1],
                              f_dict['encoder']))
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
    # Figure 3a
    methods = sorted(df_all['Dimension reduction method'].unique())
    methods = [
        # '- (d=-1) BaseNEncoder',
        '- (d=-1) HashingEncoder',
        '- (d=-1) one-hot_encoding_sparse',
        '- (d=-1) MDVEncoder',
        '- (d=-1) TargetEncoder',
        '- (d=-1) 3grams_count_vectorizer',
        '- (d=-1) jaro-winkler_SimilarityEncoder',
        '- (d=-1) levenshtein-ratio_SimilarityEncoder',
        '- (d=-1) 3gram_SimilarityEncoder',
        ][::-1]
    methods_dict = {
        '- (d=-1) one-hot_encoding_sparse': 'One-hot encoding',
        '- (d=-1) HashingEncoder': 'Hash encoding',
        '- (d=-1) BaseNEncoder': 'Binary encoding',
        '- (d=-1) TargetEncoder': 'Target encoding',
        '- (d=-1) MDVEncoder': 'MDV',
        '- (d=-1) 3gram_SimilarityEncoder': '3-gram    ',
        '- (d=-1) jaro-winkler_SimilarityEncoder': 'Jaro-winkler',
        '- (d=-1) levenshtein-ratio_SimilarityEncoder':
            'Levenshtein-\nratio      ',
        '- (d=-1) 3grams_count_vectorizer': '3-grams count vector'
        }
    palette_sns = sns.color_palette('pastel')
    palette = [
               '#bababa',
               '#bababa',
               '#bababa',
               '#bababa',
               '#bababa',
               palette_sns[0],
               palette_sns[1],
               palette_sns[2]][::-1]
    palette_dict = {method: palette[i] for i, method in enumerate(methods)}
    fontsize = 32
    clfs = sorted(df_all['clf'].unique())
    Encoders = sorted(df_all['encoder'].unique())
    score = 'score'

    # iterate over methods for each dataset
    fig, ax = plt.subplots(1, len(datasets),
                           figsize=(len(datasets)*2.5, len(methods)))
    plt.subplots_adjust(wspace=.06)
    methods_by_dataset = {dataset: 0 for dataset in datasets}
    medians_dict = {}
    rank_medians_dict = {}
    median_ohe = np.zeros(len(datasets))
    for i, dataset in enumerate(datasets):
        X = []
        av_train_time = []
        labels = []
        for method in methods:
            Encoder = method.split(' ')[-1]
            df = df_all[(df_all['dataset'] == dataset) &
                        (df_all['encoder'] == Encoder) &
                        (df_all['Dimension reduction method'] == method)]
            av_train_time.append(df['training_time'].mean())
            x = df[score].values
            labels.append(method)
            if len(x) == 0:
                X.append([])
                continue
            if len(x) > 100:
                if sum(x[0:100] == x[100:]) == 100:
                    x = x[0:100]
                else:
                    x = x[0:100]
                    # raise ValueError('Something weird happens')
            X.append(x)
            methods_by_dataset[dataset] += 1
        medianprops = dict(linestyle='-', linewidth=linewidth,
                           color='red')
        n1 = len(methods)
        n2 = len(datasets)
        bplot = ax[i].boxplot(X,
                              labels=labels, vert=False, patch_artist=True,
                              medianprops=medianprops, widths=.75, zorder=3,
                              showfliers=False)
        for box, label in zip(bplot['boxes'], labels):
            box.set(linewidth=linewidth, color='darkslateblue')
            box.set(facecolor=palette_dict[label])
        for cap in bplot['caps']:
            cap.set(linewidth=linewidth, color='darkslateblue')
        for whisker in bplot['whiskers']:
            whisker.set(linewidth=linewidth, color='darkslateblue')
        ymin, ymax = ax[i].get_ylim()
        xmin, xmax = ax[i].get_xlim()
        if (dataset == 'medical_charge') & (clf == 'Ridge'):
            ax[i].set_xlim((.63, .92))
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
        median_ohe[i] = np.median(X[6])
        medians = np.array([np.median(x) if len(x) > 0 else np.nan for x in X])
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
            if j >= 3:
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
                if j >= 3:
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

            y_i = y_f

    # average ranking
    for i in [len(datasets)-1]:
        ymin, ymax = ax[i].get_ylim()
        xmin, xmax = ax[i].get_xlim()
        average_ranking = [np.nanmean([rank_medians_dict[ds][i]
                                       for ds in datasets])
                           for i in range(len(methods))]
        for m in range(len(methods)):
            bbox = dict(boxstyle="round", fc="w", ec="0.9", alpha=0)
            plt.annotate('%.1f' % average_ranking[m],
                         xy=(1.21, 1./len(methods)*(m+.5)),
                         xycoords='axes fraction',
                         bbox=bbox, fontsize=fontsize-2,
                         ha='center', va='center')
        ax[i].axhspan(ymin, ymax, xmin=xmin_, xmax=xmax_,
                      facecolor='None',
                      edgecolor='black', linewidth=linewidth,
                      clip_on=False,
                      zorder=4)
        plt.annotate('Average ranking across datasets',
                     xy=(1.65, .45),
                     xycoords='axes fraction',
                     bbox=bbox, fontsize=fontsize,
                     ha='right', va='center', rotation=-90)
    # annotations
    for i in [0]:
        lw = 2*linewidth
        xy1 = -1.4
        xy2 = 1/len(methods)
        bbox = dict(boxstyle="round", fc="w", ec="0.9", alpha=0)
        arrowprops = dict(arrowstyle='-[, widthB=2.4', lw=lw, color='#ffaaaa')

        ax[i].annotate('Similarity\nencoding', xy=(xy1, xy2*1.5),
                       xytext=(xy1-.4, xy2*1.5),
                       xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center',
                       bbox=bbox,
                       arrowprops=arrowprops, rotation=90)
    # xticks
    for i in range(len(datasets)):
        ax[i].tick_params(axis='x', which='major',
                          labelsize=fontsize-5)
        # plt.draw()
        ax[i].set_xticklabels(ax[i].get_xticks())
        xticks = ax[i].get_xticklabels()
        # print(i, len(labels))
        # for label in labels:
        #     print(label.get_text())
        if len(xticks) > 4:
            ax[i].set_xticks(ax[i].get_xticks()[np.array([1, 3])])
            ax[i].set_xticklabels([l.get_text() for j, l
                                   in enumerate(xticks)
                                   if j in [1, 3]])
        ax[i].xaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.2f'))
        ymin, ymax = ax[i].get_ylim()
        ax[i].axvline(x=median_ohe[i], linewidth=linewidth, color='grey',
                      linestyle='--', zorder=1)
    figname = 'datasets_encoders_%s.pdf' % clf
    print('Saving: ' + figname)
    plt.savefig(os.path.join(path, figname),
                transparent=False, bbox_inches='tight', pad_inches=0.2)
    fig
