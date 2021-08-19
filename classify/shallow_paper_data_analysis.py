import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

"""
Running shallow paper analysis

This script does the data analysis for the shallow paper...
"""

# Collecting the datasets

file_dir = './apc-data/processed-5000-events/'

feature_set = []
posture_set = []
participant_ids = []

EPOCH_SIZES = [5, 15, 30, 60, 120, 180]

for epoch_size in EPOCH_SIZES:
  feature_set.append(np.load(file_dir + 'set_size_' + str(epoch_size) + '_feature_set.npy'))
  posture_set.append(np.load(file_dir + 'set_size_' + str(epoch_size) + '_feature_set_classes.npy'))
  participant_ids.append(np.load(file_dir + 'set_size_' + str(epoch_size) + '_feature_set_ids.npy'))

# Combining similar activity codes

#for epoch, dataset in enumerate(posture_set):
  #print('Original event codes for epoch ' + str(EPOCH_SIZES[epoch]))
  #print(np.unique(dataset))

for epoch, dataset in enumerate(posture_set):
  for count, value in enumerate(dataset):
    if dataset[count] == 2.1:
      dataset[count] = 2
    elif dataset[count] == 3.1:
      dataset[count] = 3
    elif dataset[count] == 3.2:
      dataset[count] = 3
    else:
      continue
  posture_set[epoch] = dataset

#for epoch, dataset in enumerate(posture_set):
  #print('New event codes for epoch ' + str(EPOCH_SIZES[epoch]))
  #print(np.unique(dataset))

# Removing specific activity codes from dataset

for count, epoch in enumerate(EPOCH_SIZES):
  #print('Number of pre event codes for epoch ' + str(epoch))
  #print(len(posture_set[count]))

  for class_to_remove in [4,5]:
    keep_idx = posture_set[count] != class_to_remove
    posture_set[count] = posture_set[count][keep_idx]
    feature_set[count] = feature_set[count][keep_idx]
    participant_ids[count] = participant_ids[count][keep_idx]

 #print('Number of post event codes for epoch ' + str(epoch))
  #print(len(posture_set[count]))

# How many participants are in the datasets

print('Number of participants')
print(len(np.unique(participant_ids[0])))

for count, epoch in enumerate(EPOCH_SIZES):

  print('Running analysis on epoch size ' + str(epoch))

  analysis_feature_set = feature_set[count].copy()
  analysis_posture_set = posture_set[count].copy()
  analysis_participant_ids = participant_ids[count].copy()

  # Setting up the printing file

  f = open("results_epoch_" + str(epoch) + ".txt", "a")

  # Converting the dataset into a df

  feature_names = [
                  'x mean',
                  'y mean',
                  'z mean',
                  'vm mean',
                  'x std',
                  'y std',
                  'z std',
                  'vm std',
                  'x med abs dev',
                  'y med abs dev',
                  'z med abs dev',
                  'vm med abs dev',
                  'x max',
                  'y max',
                  'z max',
                  'vm max',
                  'x min',
                  'y min',
                  'z min',
                  'vm min',
                  'x sig mag area',
                  'y sig mag area',
                  'z sig mag area',
                  'vm sig mag area',
                  'x energy',
                  'y energy',
                  'z energy',
                  'vm energy',
                  'x int qu range',
                  'y int qu range',
                  'z int qu range',
                  'vm int qu range',
                  'x autocorrelation',
                  'y autocorrelation',
                  'z autocorrelation',
                  'vm autocorrelation',
                  'x spec peak pos 1',
                  'x spec peak pos 2',
                  'x spec peak pos 3',
                  'x spec peak pos 4',
                  'x spec peak pos 5',
                  'x spec peak pos 6',
                  'x spec peak freq 1',
                  'x spec peak freq 2',
                  'x spec peak freq 3',
                  'x spec peak freq 4',
                  'x spec peak freq 5',
                  'x spec peak freq 6',
                  'y spec peak pos 1',
                  'y spec peak pos 2',
                  'y spec peak pos 3',
                  'y spec peak pos 4',
                  'y spec peak pos 5',
                  'y spec peak pos 6',
                  'y spec peak freq 1',
                  'y spec peak freq 2',
                  'y spec peak freq 3',
                  'y spec peak freq 4',
                  'y spec peak freq 5',
                  'y spec peak freq 6',
                  'z spec peak pos 1',
                  'z spec peak pos 2',
                  'z spec peak pos 3',
                  'z spec peak pos 4',
                  'z spec peak pos 5',
                  'z spec peak pos 6',
                  'z spec peak freq 1',
                  'z spec peak freq 2',
                  'z spec peak freq 3',
                  'z spec peak freq 4',
                  'z spec peak freq 5',
                  'z spec peak freq 6',
                  'vm spec peak pos 1',
                  'vm spec peak pos 2',
                  'vm spec peak pos 3',
                  'vm spec peak pos 4',
                  'vm spec peak pos 5',
                  'vm spec peak pos 6',
                  'vm spec peak freq 1',
                  'vm spec peak freq 2',
                  'vm spec peak freq 3',
                  'vm spec peak freq 4',
                  'vm spec peak freq 5',
                  'vm spec peak freq 6',
                  'x spec power band 1',
                  'x spec power band 2',
                  'x spec power band 3',
                  'x spec power band 4',
                  'y spec power band 1',
                  'y spec power band 2',
                  'y spec power band 3',
                  'y spec power band 4',
                  'z spec power band 1',
                  'z spec power band 2',
                  'z spec power band 3',
                  'z spec power band 4',
                  'vm spec power band 1',
                  'vm spec power band 2',
                  'vm spec power band 3',
                  'vm spec power band 4',
  ]

  print('Number of features', file=f)
  print(len(feature_names), file=f)

  # Feature transformation

  standardize = MinMaxScaler()
  analysis_feature_set_scaled = standardize.fit_transform(analysis_feature_set)
  analysis_set = pd.DataFrame(data=analysis_feature_set_scaled, columns=feature_names)

  # Removing correlated features

  correlated_features = set()
  correlation_matrix = analysis_set.corr()

  for i in range(len(correlation_matrix .columns)):
      for j in range(i):
          if abs(correlation_matrix.iloc[i, j]) > 0.8:
              colname = correlation_matrix.columns[i]
              correlated_features.add(colname)

  print('Number of correlated features:', file=f)
  print(len(correlated_features), file=f)
  #print('Correlated feature list:')
  #print(correlated_features)

  analysis_set.drop(labels=correlated_features, axis=1, inplace=True)

  # Removing quasi-constant features

  qconstant_filter = VarianceThreshold(threshold=0.01)
  qconstant_filter.fit(analysis_set)

  qconstant_columns = [column for column in analysis_set.columns
                      if column not in analysis_set.columns[qconstant_filter.get_support()]]

  print('Number of quasi-constant features:', file=f)
  print(len(qconstant_columns), file=f)
  #print('Quasi-constant feature list:')
  qconstant_columns = set(qconstant_columns)
  #print(qconstant_columns)

  analysis_set.drop(labels=qconstant_columns, axis=1, inplace=True)

  print('Prediction features:', file=f)
  print(analysis_set.columns, file=f)

  # Adding in the posture codes and participant ids

  analysis_set['posture code'] = analysis_posture_set
  analysis_set['participant id'] = analysis_participant_ids

  # Data split plots
  cmap_data = plt.cm.Paired
  cmap_cv = plt.cm.coolwarm

  def visualize_groups(classes, groups, name):
      # Visualize dataset groups
      fig, ax = plt.subplots(figsize=(10,6))
      ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
                lw=50)#, cmap=cmap_data)
      ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
                lw=50)#, cmap='Set1')
      ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
            yticklabels=['Data\ngroups', 'Data\nclasses'], xlabel="Sample index")
      plt.savefig('Data_Split_epoch_' + str(epoch) + '_.png')
      plt.close()

  visualize_groups(analysis_set['posture code'].sort_values(), analysis_set['participant id'], 'no groups')

  def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
      """Create a sample plot for indices of a cross-validation object."""

      # Generate the training/testing visualizations for each CV split
      for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
          # Fill in indices with the training/test groups
          indices = np.array([np.nan] * len(X))
          indices[tt] = 1
          indices[tr] = 0

          # Visualize the results
          ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                    c=indices, marker='_', lw=lw, cmap=cmap_cv,
                    vmin=-.2, vmax=1.2);

      # Plot the data classes and groups at the end
      ax.scatter(range(len(X)), [ii + 1.5] * len(X),
                c=y, marker='_', lw=lw, cmap=cmap_data);

      ax.scatter(range(len(X)), [ii + 2.5] * len(X),
                c=group, marker='_', lw=lw, cmap=cmap_data);

      # Formatting
      yticklabels = list(range(n_splits)) + ['class', 'group']
      ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
            xlabel='Sample index', ylabel="CV iteration",
            ylim=[n_splits+2.2, -.2]) #xlim=[0, 100]
      ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
      plt.savefig('Data_Split_KFold_epoch_' + str(epoch) + '_.png')
      plt.close()
      return ax

  fig, ax = plt.subplots(figsize=(10,6))
  #cv = KFold(n_splits=folds, random_state=None, shuffle=False)
  cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
  plot_cv_indices(cv, analysis_set.drop(labels=['posture code'], axis=1), analysis_set['posture code'], analysis_set['participant id'], ax, n_splits=10)

  # Train test split

  print('Running test-train split analysis')
  print('Running test-train split analysis', file=f)

  X_train, X_test, y_train, y_test = train_test_split(
      analysis_set.drop(labels=['posture code', 'participant id'], axis=1),
      analysis_set['posture code'],
      test_size=0.2,
      random_state=42)

  LABELS = ['Sedentary', 'Standing', 'Stepping', 'Lying']

  # Displaying class balance

  unique_classes_train = np.unique(y_train)
  unique_classes_test = np.unique(y_test)

  # Count the unique label values
  unique_train, counts_train = np.unique(y_train, return_counts=True)
  unique_test, counts_test = np.unique(y_test, return_counts=True)
  try:
    count_class_values_train = dict(zip(labels, counts_train))
    count_class_values_test = dict(zip(labels, counts_test))
  except:
    count_class_values_train = dict(zip(unique_train, counts_train))
    count_class_values_test = dict(zip(unique_test, counts_test))

  print('Train Classes', file=f)
  print(count_class_values_train, file=f)
  print('--------------', file=f)
  print('Test Classes', file=f)
  print(count_class_values_test, file=f)
  print('--------------', file=f)

  # Transform list

  LDA = LinearDiscriminantAnalysis(n_components=2)
  LDA.fit(X_train, y_train)

  # Model list

  models = (KNeighborsClassifier(n_neighbors=10),
            KNeighborsClassifier(n_neighbors=10),
            svm.SVC(kernel='rbf', gamma=1, C=1),
            RandomForestClassifier(n_estimators=30),
            ExtraTreesClassifier(n_estimators=30),
            LogisticRegression(solver='sag', max_iter=100, multi_class='ovr'),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            )

  model_titles = ('KNN',
                  'KNN (LDA)',
                  'SVM',
                  'RF',
                  'Extra Trees RF',
                  'Log Reg',
                  'Naive Bayes',
                  'Quad Discriminant Analysis',
                  )

  # Assess model

  fit_models = []
  for clf, title in zip(models, model_titles):
    print('Fitting model ' + title)
    if title == 'KNN (LDA)':
        clf.fit(LDA.transform(X_train), y_train)
    else:
        clf.fit(X_train, y_train)
    fit_models.append(clf)

  for clf, title in zip(fit_models, model_titles):
    print('Evaluating model ' + title)
    if title == 'KNN (LDA)':
      #acc = clf.score(X_test, y_test)
      predictions = clf.predict(LDA.transform(X_test))
    else:
      predictions = clf.predict(X_test)
    print('Classification report (train test split) - ' + title, file=f)
    print(classification_report(y_test, predictions.astype(int)), file=f)

    matrix = metrics.confusion_matrix(y_test, predictions.astype(int), normalize ='true')
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion_matrix_' + title + '_epoch_' + str(epoch) + '.png')
    plt.close()

  # Creating a new training set for the K Fold

  X = analysis_set.drop(labels=['posture code', 'participant id'], axis=1)
  y = analysis_set['posture code']

  print('Running K Fold analysis')
  print('Running K Fold analysis', file=f)

  # Class balance of entire dataset

  unique, counts = np.unique(analysis_set['posture code'], return_counts=True)
  try:
    count_class_values = dict(zip(labels, counts))
  except:
    count_class_values = dict(zip(unique, counts))

  print('Train Classes', file=f)
  print(count_class_values, file=f)
  print('--------------', file=f)

  # Transform list for K Fold

  LDA = LinearDiscriminantAnalysis(n_components=2)
  LDA.fit(X_train, y_train)

  # Assess models K Fold

  cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

  fit_models = []
  for clf, title in zip(models, model_titles):
      print('Fitting model ' + title)
      if title == 'KNN (LDA)':
          clf.fit(LDA.transform(X), y)
      else:
          clf.fit(X, y)
      fit_models.append(clf)

  for clf, title in zip(fit_models, model_titles):
    print('Evaluating model ' + title)
    if title == 'KNN (LDA)':
      scores = cross_val_score(clf, LDA.transform(X), y, cv=cv)
      y_pred = cross_val_predict(clf, LDA.transform(X), y, cv=cv)
    else:
      scores = cross_val_score(clf, X, y, cv=cv)
      y_pred = cross_val_predict(clf, X, y, cv=cv)

    print('Classification report (K Fold) - ' + title, file=f)
    print(classification_report(y, y_pred), file=f)

    conf_mat = confusion_matrix(y, y_pred, normalize ='true')
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion_matrix_' + title + '_KFlod_epoch_' + str(epoch) + '.png')
    plt.close()
  
  print('Finished analysis on epoch size ' + str(epoch))
  f.close()