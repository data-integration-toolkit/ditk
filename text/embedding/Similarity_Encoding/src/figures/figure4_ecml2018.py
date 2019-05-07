import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

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

datasets = [
            'employee_salaries',
            'docs_payments',
            'midwest_survey',
            'traffic_violations',
            'road_safety',
            'beer_reviews'
            ]
# 'indultos_espana', 'sydney_cycling'
linewidth = 1
results_path = os.path.join(get_data_folder(), 'results', 'ecml2018')
path = os.path.join(CE_HOME, 'submissions', 'ecml2018')
##############################################################################

datasets_name = {'employee_salaries': 'employee\nsalaries',
                 'traffic_violations': 'traffic\nviolations',
                 'beer_reviews': 'beer\nreviews',
                 'midwest_survey': 'midwest\nsurvey',
                 'docs_payments': 'open\npayments',
                 'medical_charge': 'medical\ncharges',
                 'road_safety': 'road\nsafety'
                 }

clfs = ['Ridge', 'GradientBoosting']
# clfs = ['GradientBoosting']
dimSE_nored = {dataset: 0 for dataset in datasets}
for clf in clfs:
    plt.close('all')
    df_all = pd.DataFrame()
    score_type = {}
    for dataset in datasets:
        data = Data(dataset)
        if dataset in ['docs_payments', 'beer_reviews2', 'traffic_violations']:
            n_rows = 100000  # -1 if using all rows for prediction
        elif dataset in ['beer_reviews', 'road_safety']:
            n_rows = 10000
        else:
            n_rows = -1
        conditions = {'dataset': data.name,
                      'n_splits': 100,
                      'test_size': .2,
                      'n_rows': n_rows,
                      'str_preprocess': True,
                      'encoder': ['3gram_SimilarityEncoder',
                                  'one-hot_encoding_sparse'],
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
        for f in files2:
            f_dict = read_json(f)
            df = pd.DataFrame(f_dict['results'])
            # df = df.drop_duplicates(subset=df.columns[1:])
            df['dataset'] = data.name
            df['clf'] = f_dict['clf'][0]
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
            df['Dimension reduction method'] = dim_red_method
            df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
        # if clf == 'Ridge':
        max_f = df_all[
            (df_all['fold'] == 1) &
            (df_all['dataset'] == dataset) &
            (df_all['Dimension reduction method'] ==
             '- (d=-1) one-hot_encoding_sparse')
            ]['n_train_features'].max()
        min_f = df_all[
            (df_all['fold'] == 1) &
            (df_all['dataset'] == dataset) &
            (df_all['Dimension reduction method'] ==
             'MostFrequentCategories (d=30) 3gram_SimilarityEncoder')
            ]['n_train_features'].min()
        print(min_f, max_f)
        if min_f > 0 and max_f > 0:
            dimSE_nored[dataset] = max_f-min_f+30
    df_all[df_all['dataset'] == 'docs_payments']
    # Figure 3a
    methods = sorted(df_all['Dimension reduction method'].unique())
    methods = [
        '- (d=-1) one-hot_encoding_sparse',
        'RandomProjectionsGaussian (d=300) one-hot_encoding_sparse',
        'RandomProjectionsGaussian (d=100) one-hot_encoding_sparse',
        'RandomProjectionsGaussian (d=30) one-hot_encoding_sparse',
        'KMeans_clustering (d=300) 3gram_SimilarityEncoder',
        'KMeans_clustering (d=100) 3gram_SimilarityEncoder',
        'KMeans_clustering (d=30) 3gram_SimilarityEncoder',
        '- (d=-1) 3gram_SimilarityEncoder',
        'KMeans (d=300) 3gram_SimilarityEncoder',
        'KMeans (d=100) 3gram_SimilarityEncoder',
        'KMeans (d=30) 3gram_SimilarityEncoder',
        'MostFrequentCategories (d=300) 3gram_SimilarityEncoder',
        'MostFrequentCategories (d=100) 3gram_SimilarityEncoder',
        'MostFrequentCategories (d=30) 3gram_SimilarityEncoder',
        'RandomProjectionsGaussian (d=300) 3gram_SimilarityEncoder',
        'RandomProjectionsGaussian (d=100) 3gram_SimilarityEncoder',
        'RandomProjectionsGaussian (d=30) 3gram_SimilarityEncoder'][::-1]
    methods_dict = {
        '- (d=-1) one-hot_encoding_sparse': 'Full',
        'RandomProjectionsGaussian (d=300) one-hot_encoding_sparse': 'd=300',
        'RandomProjectionsGaussian (d=100) one-hot_encoding_sparse': 'd=100',
        'RandomProjectionsGaussian (d=30) one-hot_encoding_sparse': 'd=30',
        'KMeans_clustering (d=300) 3gram_SimilarityEncoder': 'd=300',
        'KMeans_clustering (d=100) 3gram_SimilarityEncoder': 'd=100',
        'KMeans_clustering (d=30) 3gram_SimilarityEncoder': 'd=30',
        '- (d=-1) 3gram_SimilarityEncoder': 'Full',
        'KMeans (d=300) 3gram_SimilarityEncoder': 'd=300',
        'KMeans (d=100) 3gram_SimilarityEncoder': 'd=100',
        'KMeans (d=30) 3gram_SimilarityEncoder': 'd=30',
        'MostFrequentCategories (d=300) 3gram_SimilarityEncoder': 'd=300',
        'MostFrequentCategories (d=100) 3gram_SimilarityEncoder': 'd=100',
        'MostFrequentCategories (d=30) 3gram_SimilarityEncoder': 'd=30',
        'RandomProjectionsGaussian (d=300) 3gram_SimilarityEncoder': 'd=300',
        'RandomProjectionsGaussian (d=100) 3gram_SimilarityEncoder': 'd=100',
        'RandomProjectionsGaussian (d=30) 3gram_SimilarityEncoder': 'd=30'}
    palette = ['#bababa',
               '#2eb82e', '#70db70', '#c2f0c2',
               '#ff6de7', '#ff96ee', '#ffbff5',
               sns.color_palette('pastel')[2],
               '#ffc044', '#ffd27a', '#ffdfa0',
               '#005ce6', '#4d94ff', '#b3d1ff',
               '#2eb82e', '#70db70', '#c2f0c2'][::-1]
    palette_dict = {method: palette[i] for i, method in enumerate(methods)}
    fontsize = 26
    clfs = sorted(df_all['clf'].unique())
    distances = sorted(df_all['encoder'].unique())
    score = 'score'

    # iterate over methods for each dataset
    fig, ax = plt.subplots(
        1, len(datasets), figsize=(len(datasets)*2.5, len(methods)/2.))
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
            distance = method.split(' ')[-1]
            df = df_all[(df_all['dataset'] == dataset) &
                        (df_all['encoder'] == distance) &
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
                    print('something weird happens')
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
        if (dataset == 'beer_reviews') & (clf == 'GradientBoosting'):
            ax[i].set_xlim([0.1, 0.83])
        if (dataset == 'traffic_violations') & (clf == 'GradientBoosting'):
            ax[i].set_xlim([0.7254, 0.798])
        if (dataset == 'employee_salaries') & (clf == 'Ridge'):
            ax[i].set_xlim([0.48, .93])
        ax[i].tick_params(axis='both', which='major', labelsize=fontsize-4)
        ax[i].set_xlabel(datasets_name[dataset], labelpad=10,
                         fontsize=fontsize, rotation='horizontal')
        if i == 0:
            methods_vals = [methods_dict[method] for method in methods]
            ax[0].set_yticklabels(methods_vals, ha='left')
            # align yticklabels to the left
            plt.draw()  # because get_window_extent needs a renderer to work
            yax = ax[i].get_yaxis()
            pad = max(T.label.get_window_extent().width
                      for T in yax.majorTicks) + 6
            yax.set_tick_params(pad=pad)
        else:
            ax[i].set_yticklabels([])
        # putting an asterisc on the prediction with the maximum median
        median_ohe[i] = np.median(X[-1])
        medians = np.array([np.median(x) if len(x) > 0 else np.nan for x in X])
        medians_dict[dataset] = medians
        max_median_method = np.argsort(medians)[::-1]
        rank_medians_dict[dataset] = ss.rankdata(-medians)
        rank_medians_dict[dataset][np.isnan(medians)] = np.nan
        y_i = ymin
        delta = (ymax-ymin)/len(methods)
        for j, method in enumerate(methods):
            y_f = y_i + delta

            if j >= 10:
                if j/2.0 == int(j/2):
                    ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#ffffff')
                else:
                    ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#f4f4f4')
            else:
                if j/2.0 == int(j/2):
                    ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#fff2f2')
                else:
                    ax[i].axhspan(y_i, y_f, xmin=0, xmax=1, color='#ffe8e8')

            if i == len(datasets)-1:
                xmin_, xmax_ = 1.05, 1.37
                if j >= 10:
                    if j/2.0 == int(j/2):
                        ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                      color='#ffffff', clip_on=False, zorder=2)
                    else:
                        ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                      color='#f4f4f4', clip_on=False, zorder=2)
                else:
                    if j/2.0 == int(j/2):
                        ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                      color='#fff2f2', clip_on=False, zorder=2)
                    else:
                        ax[i].axhspan(y_i, y_f, xmin=xmin_, xmax=xmax_,
                                      color='#ffe8e8', clip_on=False, zorder=2)
            y_i = y_f

        # cardinality full
        xmin, xmax = ax[i].get_xlim()
        ax[i].text(xmin + (xmax-xmin)*.5, -3,
                   ('(k=%d)' % dimSE_nored[dataset]),
                   fontsize=fontsize-4, ha='center', va='top', color='gray',
                   weight='normal')
        if i == 0:
            ax[i].text(xmin - (xmax-xmin)*.6, -3.3,
                       'Cardinality of\ncategorical variable',
                       fontsize=fontsize-4, ha='center', va='center',
                       color='gray', weight='normal')
        if i != 0:
            ax[i].tick_params(axis='y', which='both', length=0)

    # average ranking
    for i in [len(datasets)-1]:
        average_ranking = [np.nanmean([rank_medians_dict[ds][i]
                                       for ds in datasets])
                           for i in range(len(methods))]
        for m in range(len(methods)):
            ax[i].text(xmax+(xmax-xmin)*.35, m+.9,
                       '%.1f' % average_ranking[m], fontsize=fontsize-3,
                       ha='right', va='center', color='black', zorder=3,
                       weight='normal')
        ax[i].axhspan(ymin, ymax, xmin=xmin_, xmax=xmax_, facecolor='None',
                      edgecolor='black', linewidth=linewidth, clip_on=False,
                      zorder=4)
        ax[i].text(xmax+(xmax-xmin)*.57, (ymax+ymin)/2,
                   'Average ranking across datasets', fontsize=fontsize,
                   ha='right', va='center', color='black', rotation=-90,
                   weight='normal')
    # annotations
    for i in [0]:
        lw = 2*linewidth
        xy1, xy2 = -1.5, 1/len(methods)
        xytext1, xytext2 = xy1 - .3, 1/(len(methods))
        bbox = dict(boxstyle="round", fc="w", ec="0.9", alpha=0)
        arrowprops = dict(arrowstyle='-[, widthB=1.9', lw=lw, color='#a3a3a3')
        ax[i].annotate('One-hot\nencoding', xy=(xy1, xy2*15),
                       xytext=(xytext1, xytext2*15),
                       xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', bbox=bbox,
                       arrowprops=arrowprops, rotation=90)

        arrowprops = dict(arrowstyle='-[, widthB=4.9', lw=lw,
                          color=sns.color_palette('pastel')[2])
        ax[i].annotate('3-gram similarity\nencoding', xy=(xy1, xy2*5),
                       xytext=(xytext1, xytext2*5),
                       xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', bbox=bbox,
                       arrowprops=arrowprops, rotation=90)
        lw = 2*linewidth
        arrowstyle = '-[, widthB=1.6'
        arrowprops = dict(arrowstyle=arrowstyle, lw=lw)
        xy1, xy2 = -.55, 1/len(methods)
        xytext1, xytext2 = xy1-.52, 1/(len(methods))
        xycoords = 'axes fraction'
        ha = 'center'
        va = 'center'
        rotation = 0
        arrowprops = dict(arrowstyle=arrowstyle, lw=lw, color='#2eb82e')
        ax[i].annotate('Random\nprojections',
                       xy=(xy1, xy2*1.5), xytext=(xytext1, xytext2*1.5),
                       xycoords=xycoords, fontsize=fontsize-2, ha=ha, va=va,
                       bbox=bbox, arrowprops=arrowprops, rotation=rotation)
        arrowprops = dict(arrowstyle=arrowstyle, lw=lw, color='#005ce6')
        ax[i].annotate('Most\nfrequent\ncategories',
                       xy=(xy1, xy2*4.5), xytext=(xytext1, xytext2*4.5),
                       xycoords=xycoords, fontsize=fontsize-2, ha=ha, va=va,
                       bbox=bbox, arrowprops=arrowprops, rotation=rotation)
        arrowprops = dict(arrowstyle=arrowstyle, lw=lw, color='#ffc044')
        ax[i].annotate(' K-means ',
                       xy=(xy1, xy2*7.5), xytext=(xytext1, xytext2*7.5),
                       xycoords=xycoords, fontsize=fontsize-2, ha=ha, va=va,
                       bbox=bbox, arrowprops=arrowprops, rotation=rotation)
        arrowprops = dict(arrowstyle=arrowstyle, lw=lw, color='#ff6de7')
        ax[i].annotate('Deduplication\nwith K-means',
                       xy=(xy1, xy2*11.5), xytext=(xytext1-.1, xytext2*11.5),
                       xycoords=xycoords, fontsize=fontsize-2, ha=ha, va=va,
                       bbox=bbox, arrowprops=arrowprops, rotation=rotation)
        arrowprops = dict(arrowstyle=arrowstyle, lw=lw, color='#2eb82e')
        ax[i].annotate('Random\nprojections',
                       xy=(xy1, xy2*14.5), xytext=(xytext1, xytext2*14.5),
                       xycoords=xycoords, fontsize=fontsize-2, ha=ha, va=va,
                       bbox=bbox, arrowprops=arrowprops, rotation=rotation)
    for i in range(len(datasets)):
        ymin, ymax = ax[i].get_ylim()
        ax[i].axvline(x=median_ohe[i], linewidth=linewidth, color='grey',
                      linestyle='--', zorder=1)
    # plt.suptitle('Scores by dataset with different ' +
    #              'dimensionality reduction methods. Classifier: %s' % clf,
    #              fontsize=fontsize, weight='bold', y=.95, x=0, ha='left')
    figname = ('datasets_dimension-reduction_' + clf + '.pdf')
    print('Saving: ' + figname)
    plt.savefig(os.path.join(path, figname),
                transparent=False, bbox_inches='tight', pad_inches=0.2)
