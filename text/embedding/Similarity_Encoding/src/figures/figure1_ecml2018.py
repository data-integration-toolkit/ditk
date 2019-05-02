import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from datasets import Data, get_data_folder
from fns_categorical_encoding import file_meet_conditions2, read_json
from fns_figures_dataset import set_list


# PARAMETERS ##################################################################
path = os.path.join(CE_HOME, 'submissions', 'ecml2018')
sns.set_style("ticks")
dataset_cm = {'docs_payments': 0,
              'midwest_survey': 1,
              'medical_charge': 2,
              'traffic_violations': 3,
              'beer_reviews': 4,
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
datasets = ['medical_charge', 'employee_salaries',
            'docs_payments', 'midwest_survey',
            'traffic_violations',
            'road_safety', 'beer_reviews']
datasets = datasets[::-1]
linewidth = 1.4
n_splits = 100
grid = np.linspace(100, 300000, 100)

datasets_name = {'employee_salaries': 'employee salaries',
                 'traffic_violations': 'traffic violations',
                 'beer_reviews': 'beer reviews',
                 'midwest_survey': 'midwest survey',
                 'docs_payments': 'open payments',
                 'medical_charge': 'medical charges',
                 'road_safety': 'road safety'
                 }
# ##############################################################################
# # Table 1: All datasets; Size
#
# df = pd.DataFrame()
# for dataset in datasets[::-1]:
#     print(dataset)
#     data = Data(dataset).get_df()
#     if dataset in ['docs_payments', 'crime_data', 'beer_reviews2',
#                    'traffic_violations']:
#         n_rows = 100000  # -1 if using all rows for prediction
#     elif dataset in ['beer_reviews', 'road_safety']:
#         n_rows = 10000
#     else:
#         n_rows = -1
#     if dataset in ['adult', 'adult2', 'adult3']:
#         typo_prob = .1
#     else:
#         typo_prob = 0
#     data.df = data.df.sample(frac=1, random_state=5
#                              ).reset_index(drop=True)[:n_rows]
#     if data.clf_type == 'regression':
#         score_type = 'r2'
#     if data.clf_type == 'binary_clf':
#         score_type = 'average precision'
#     if data.clf_type == 'multiclass_clf':
#         score_type = 'accuracy'
#     df['score_type'] = data.clf_type
#     y_col = [col for col in data.col_action if data.col_action[col] is 'y'][0]
#     sm_col = [col for col in data.col_action
#               if data.col_action[col] is 'se'][0]
#     num_cat = len(data.df[sm_col].unique())
#     hf_cat = data.df[sm_col].value_counts()[0]
#     lf_cat = data.df[sm_col].value_counts()[-1]
#     if dataset[:5] == 'adult':
#         dataset = ('adult (%.0f%% typos)' % (typo_prob*100))
#     data_dict = {'Dataset': datasets_name[dataset],
#                  #  '# columns': '%d' % data.df.shape[1],
#                  '# rows': '%.1E' % data.df.shape[0],
#                  #  'Prediction \\variable': y_col,
#                  #  'Similarity \\ encoding variable': sm_col,
#                  '# categories': '%d' % num_cat,
#                  'Highest frequency category': '%d' % hf_cat,
#                  'Lowest frequency category': '%d' % lf_cat,
#                  'Prediction type': data.clf_type}
#     df = df.append(data_dict, ignore_index=True)
#     del data
#
# # DataFrame to latex
# df = df.reindex_axis(data_dict.keys(), axis=1)
# df = df.set_index('Dataset')
# df_latex = df.to_latex(bold_rows=True)
# df_latex2 = df_latex[df_latex.find('midrule')+8:df_latex.find('bottomrule')-1]
# table_file = os.path.join(path, 'table_df_datasets.txt')
# with open(table_file, 'w') as text_file:
#     text_file.write(df_latex2)
#
# ##############################################################################
# Figure 1: All datasets; Categorical variable of interest;
#           Evolution of the number of categories.

figname = 'datasets_unique_categories.pdf'
palette = sns.hls_palette(7, l=0.5, s=.5)

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

params = {'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'mathtext.bf': 'serif:bold',
          'mathtext.it': 'serif:italic',
          'mathtext.sf': 'sans\\-serif',
          'font.family': 'serif',
          'font.serif': "Times New Roman",  # or "Times"
          'text.latex.preamble':
          [r'\usepackage{siunitx}', r'\usepackage{amsmath}',
           r'\usepackage{libertine}', r'\usepackage[libertine]{newtxmath}']}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(9, 6))
fontsize = 25
for dataset in datasets:
    print(dataset)
    data = Data(dataset).get_df(preprocess_df=False)
    data.df = data.df.sample(frac=1, random_state=5).reset_index(drop=True)
    cat_variable = [x for x in data.col_action if data.col_action[x] is 'se']
    nrows_log10 = np.log10(data.df.shape[0])
    X = np.logspace(2, int(nrows_log10), int(nrows_log10)-1)
    X = np.append(X, pow(10, nrows_log10))
    Y = [len(np.unique(data.df[cat_variable].astype(str).values[:int(x)]))
         for x in X]
    ax.plot(X, Y, color=palette[dataset_cm[dataset]],
            linewidth=2,
            marker=list(markers)[dataset_cm[dataset]], markersize=10,
            zorder=3)
    del data

plt.savefig(os.path.join(path, figname),
            transparent=False, bbox_inches='tight', pad_inches=0.2)
labels = [datasets_name[ds] for ds in datasets]
for i in range(len(labels)):
    if labels[i][:5] == 'adult':
        labels[i] = 'adult (10% typos)'
leg = plt.legend(labels,
                 loc='upper left', bbox_to_anchor=(-.02, 1.03), ncol=1,
                 fontsize=fontsize-6, labelspacing=.2)
leg.set_zorder(2)
# leg.set_title('Dataset', prop={'size': 18, 'weight': 'bold'})
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Number of rows', fontsize=fontsize)
ax.set_ylabel('Number of categories', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
# plt.grid(False, color='gray')
xticklabels = [t.get_text() for t in ax.get_xticklabels()]
xticks = [t for t in ax.get_xticks()]
for i in range(len(xticks)):
    if xticklabels[i] is '':
        continue
    if (xticks[i] >= 1e3) & (xticks[i] < 1e6):
        xticklabels[i] = str('%.0fk' % (xticks[i]/1e3))
    if xticks[i] < 1e3:
        xticklabels[i] = str('%.0f' % (xticks[i]))
    if xticks[i] >= 1e6:
        xticklabels[i] = str('%.0fM' % (xticks[i]/1e6))
ax.set_xticklabels(xticklabels)
yticklabels = [t.get_text() for t in ax.get_yticklabels()]
yticks = [t for t in ax.get_yticks()]
for i in range(len(yticks)):
    if yticklabels[i] is '':
        continue
    if (yticks[i] >= 1e3) & (yticks[i] < 1e6):
        yticklabels[i] = (str(int(yticks[i]))[:-3] + ' ' +
                          str(int(yticks[i]))[-3:])
    else:
        yticklabels[i] = str(int(yticks[i]))
ax.set_yticklabels(yticklabels)

print('Saving: ' + figname)
plt.savefig(os.path.join(path, figname),
            transparent=False, bbox_inches='tight', pad_inches=0.2)
