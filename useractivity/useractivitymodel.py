from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

def load_dataset_acc(path, y):
    data = list();
    path = path.strip('/');
    acc_directory = path
    for name in listdir(acc_directory):
        filename = acc_directory + '/' + name;
        df = pd.read_csv(filename, header=0, names=['time', 'AX', 'AY', 'AZ']);
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['activity'] = y
        data.append(df)

    acc = pd.concat(data, axis=0, ignore_index=True);
    return acc;


def load_dataset_gyro(path, y):
    data = list();
    path = path.strip('/');
    gyro_directory = path
    for name in listdir(gyro_directory):
        filename = gyro_directory + '/' + name;
        df = pd.read_csv(filename, header=0, names=['time', 'GX', 'GY', 'GZ']);
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['activity'] = y
        data.append(df)

    gyro = pd.concat(data, axis=0, ignore_index=True);
    return gyro;


def load_dataset_pressure_right(path, y):
    data = list();
    path = path.strip('/');
    right_directory = path
    for name in listdir(right_directory):
        filename = right_directory + '/' + name;
        df = pd.read_csv(filename, header=0, names=['time', 'PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'PR7', 'PR8']);
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['activity'] = y
        data.append(df)

    right = pd.concat(data, axis=0, ignore_index=True);
    return right;


def load_dataset_pressure_left(path, y):
    data = list();
    left_directory = path.strip('/');
    for name in listdir(left_directory):
        filename = left_directory + '/' + name;
        df = pd.read_csv(filename, header=0, names=['time', 'PL1', 'PL2', 'PL3', 'PL4', 'PL5', 'PL6', 'PL7', 'PL8']);
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['activity'] = y
        data.append(df)

    left = pd.concat(data, axis=0, ignore_index=True);
    return left;


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0);
    sigma = np.std(dataset, axis=0);
    return (dataset - mu) / sigma;


def plot_pressure_left(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:5000]
        plt.title(data.activity.iloc[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['PL1'], label='PL1')
        plt.plot(data.index, data['PL2'], label='PL2')
        plt.plot(data.index, data['PL3'], label='PL3')
        plt.plot(data.index, data['PL4'], label='PL4')
        plt.plot(data.index, data['PL5'], label='PL5')
        plt.plot(data.index, data['PL6'], label='PL6')
        plt.plot(data.index, data['PL7'], label='PL7')
        plt.plot(data.index, data['PL8'], label='PL8')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def plot_acc(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:5000]
        plt.title(data.activity[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['AX'], label='AX')
        plt.plot(data.index, data['AY'], label='AY')
        plt.plot(data.index, data['AZ'], label='AZ')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def plot_gyro(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:2000]
        plt.title(data.activity[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['GX'], label='GX')
        plt.plot(data.index, data['GY'], label='GY')
        plt.plot(data.index, data['GZ'], label='GZ')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def plot_pressure_right(dataset):
    plt.figure()
    start = 321
    for i in range(len(dataset)):
        plt.subplot(start)
        data = dataset[i][:3000]
        plt.title(data.activity.iloc[0])
        data = data.set_index(['time'])
        plt.plot(data.index, data['PR1'], label='PR1')
        plt.plot(data.index, data['PR2'], label='PR2')
        plt.plot(data.index, data['PR3'], label='PR3')
        plt.plot(data.index, data['PR4'], label='PR4')
        plt.plot(data.index, data['PR5'], label='PR5')
        plt.plot(data.index, data['PR6'], label='PR6')
        plt.plot(data.index, data['PR7'], label='PR7')
        plt.plot(data.index, data['PR8'], label='PR8')
        plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().xaxis.set_visible(False)
        start = start + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def load_dataset():
    dataset = list();
    dataset_pressureleft = list();
    df = load_dataset_pressure_left('dataset/running/left', 'running');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressureleft.append(df);

    df = load_dataset_pressure_left('dataset/sitting/left', 'sitting');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressureleft.append(df);

    df = load_dataset_pressure_left('dataset/standing/left', 'standing');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressureleft.append(df);

    df = load_dataset_pressure_left('dataset/walking/left', 'walking');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressureleft.append(df);

    df = load_dataset_pressure_left('dataset/astair/left', 'ascendingstair');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressureleft.append(df);

    df = load_dataset_pressure_left('dataset/dstair/left', 'descendingstair');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressureleft.append(df);


    dataset_pressure_right = list();
    df = load_dataset_pressure_right('dataset/running/right', 'running');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressure_right.append(df);

    df = load_dataset_pressure_right('dataset/sitting/right', 'sitting');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressure_right.append(df);

    df = load_dataset_pressure_right('dataset/standing/right', 'standing');
    df = df[(df[df.columns.difference(['time','activity'])] >= 0).all(1)]
    dataset_pressure_right.append(df);

    df = load_dataset_pressure_right('dataset/walking/right', 'walking');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressure_right.append(df);

    df = load_dataset_pressure_right('dataset/astair/right', 'ascendingstair');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressure_right.append(df);

    df = load_dataset_pressure_right('dataset/dstair/right', 'descendingstair');
    df = df[(df[df.columns.difference(['time', 'activity'])] >= 0).all(1)]
    dataset_pressure_right.append(df);

    dataset_acc = list();
    dataset_acc.append(load_dataset_acc('dataset/running/acc', 'running'));
    dataset_acc.append(load_dataset_acc('dataset/sitting/acc', 'sitting'));
    dataset_acc.append(load_dataset_acc('dataset/standing/acc', 'standing'));
    dataset_acc.append(load_dataset_acc('dataset/walking/acc', 'walking'));
    dataset_acc.append(load_dataset_acc('dataset/astair/acc', 'ascendingstair'));
    dataset_acc.append(load_dataset_acc('dataset/dstair/acc', 'descendingstair'));

    dataset_gyro = list();
    dataset_gyro.append(load_dataset_gyro('dataset/running/gyro', 'running'));
    dataset_gyro.append(load_dataset_gyro('dataset/sitting/gyro', 'sitting'));
    dataset_gyro.append(load_dataset_gyro('dataset/standing/gyro', 'standing'));
    dataset_gyro.append(load_dataset_gyro('dataset/walking/gyro', 'walking'));
    dataset_gyro.append(load_dataset_gyro('dataset/astair/gyro', 'ascendingstair'));
    dataset_gyro.append(load_dataset_gyro('dataset/dstair/gyro', 'descendingstair'));

    dataset.append(dataset_pressureleft);
    dataset.append(dataset_pressure_right);
    dataset.append(dataset_acc);
    dataset.append(dataset_gyro);

    return dataset;

def plot_dataset(dataset):
    plot_pressure_left(dataset[0]);
    plot_pressure_right(dataset[1]);
    plot_acc(dataset[2]);
    plot_gyro(dataset[3]);

def synchronize_data(dataset):
    featureslist = list();
    for i in range(len(dataset[0])):
        datalist = dataset[0][i]
        activity = datalist['activity'].iloc[0]
        datalist = datalist.resample('1S', on='time').mean();
        datalist = datalist.reset_index();
        datalist['minsec'] = datalist['time'].dt.strftime('%M:%S');
        datalist.set_index(['minsec'])
        for j in range(1, len(dataset)):
            temp = dataset[j][i];
            temp = temp.resample('1S', on='time').mean();
            temp = temp.reset_index();
            temp['minsec'] = temp['time'].dt.strftime('%M:%S');
            temp = temp.drop(['time'], axis=1)
            temp.set_index(['minsec']);
            datalist = pd.merge(left=datalist, right=temp, left_on='minsec', right_on='minsec');


        datalist['activity'] = activity
        datalist['PL'] = datalist[['PL1', 'PL2', 'PL3', 'PL4', 'PL5', 'PL6', 'PL7', 'PL8']].mean(
            axis=1, skipna=True);
        datalist['AR'] = np.sqrt(np.power(datalist['AX'],2) + np.power(datalist['AY'],2) + np.power(datalist['AZ'], 2));
        datalist['GR'] = np.sqrt(np.power(datalist['GX'],2) + np.power(datalist['GY'],2) + np.power(datalist['GZ'], 2));
        datalist['PR'] = datalist[['PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'PR7', 'PR8']].mean(
            axis=1, skipna=True);
        datalist = datalist.dropna().drop(columns=['GX','GY','GZ','AX','AY','AZ','PL1', 'PL2', 'PL3', 'PL4', 'PL5', 'PL6', 'PL7', 'PL8','PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'PR7', 'PR8'], axis=1)
        featureslist.append(datalist);
    return featureslist;

def build_features(featurelist):
    columns = ['PLMEAN','PLSTD','PLSKEW','PLMIN','PLMAX','PLMEDIAN','PLK',
               'PRMEAN','PRSTD','PRSKEW','PRMIN','PRMAX','PRMEDIAN','PRK',
               'AMEAN','ASTD','ASKEW','AMIN','AMAX','AMEDIAN','AK',
               'GMEAN','GSTD','GSKEW','GMIN','GMAX','GMEDIAN','GK',
               'activity'];
    df = pd.DataFrame();
    datafinal = list();
    print(featurelist)
    print(len(featurelist))
    for i in range(len(featurelist)):
        newlist = list();
        data = featurelist[i] = featurelist[i].sort_values(by=['time']);
        data = data.drop(['minsec'],axis=1)
        activity = data['activity'].iloc[0]
        datalist = data.resample('2S', on='time').apply(np.nanmean);
        datalist['activity'] = activity
        newlist.append(datalist);

        datalist = data.resample('2S', on='time').apply(np.nanstd);
        newlist.append(datalist);


        datalist = data.resample('2S', on='time').apply(np.nanmin);
        newlist.append(datalist);

        datalist = data.resample('2S', on='time').apply(np.nanmax);
        newlist.append(datalist);

        datalist = data.resample('2S', on='time').apply(np.nanmedian);
        newlist.append(datalist);


        datafinal.append(newlist)


    columnList = ['MEAN','STD','MIN','MAX','MEDIAN']
    for i in range(0,len(datafinal)):
        result = pd.DataFrame()
        data = datafinal[i][0];
        column = {}
        for k in range(len(data.columns)):
            if data.columns[k] == 'time' or data.columns[k] == 'activity':
                column[data.columns[k]] = data.columns[k]
            else:
                column[data.columns[k]] = data.columns[k] + columnList[0];
        data.rename(columns=column,inplace=True)
        for j in range(1, len(datafinal[i])):
            dataRight =  datafinal[i][j]
            if 'time' in dataRight.columns:
                dataRight = dataRight.drop(['time'], axis=1)
            column = {}

            for p in range(len(dataRight.columns)):
                if dataRight.columns[p] == 'time' or dataRight.columns[p] == 'activity':
                    column[dataRight.columns[p]] = dataRight.columns[p]
                else:
                    column[dataRight.columns[p]] = dataRight.columns[p] + columnList[j];
            dataRight.rename(columns=column,inplace=True)

            if result.empty:
                result = pd.merge(left=data, right=dataRight, left_on='time', right_on='time', how='inner').reset_index();
            else:
                result = result.set_index(['time'])
                result = pd.merge(left=result, right=dataRight, left_on='time', right_on='time',
                                      how='inner').reset_index();
        print(result)
        if df.empty:
            result = result.dropna()
            df = result
        else:
            result = result.dropna()
            df = df.append(result,ignore_index=True)
        print('Feature building: ', (i * 100 // 6), '% complete');
    df = df.drop(['time'], axis=1)
    df = df.drop(['activity_x'], axis=1)
    df = df.drop(['activity_y'], axis=1)
    return df;

def predictWithFeatureSelection(X,y,topN,size):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0);

    clf = ExtraTreesClassifier(n_estimators=5000)
    clf = clf.fit(X, y)

    y_pred = clf.predict(X_test);
    cnfMatrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnfMatrix)
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    print('Feature Importances')
    print(clf.feature_importances_)
    for feature in zip(X.columns, clf.feature_importances_):
        print(feature)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            clf.estimators_],
                                           axis=0)

    XFeatures = list();


    model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=topN)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test);

    print(model.get_support(indices=True))
    for feature_list_index in model.get_support(indices=True):
        XFeatures.append(X.columns[feature_list_index])

    print('Selected Features')
    print(XFeatures);

    sc = StandardScaler();
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg', C=4);
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test);
    cnfMatrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnfMatrix)
    report = metrics.classification_report(y_test, y_pred)
    print(report)


def predictWithFeatureSelectionSVM(X,y,topN,size,cvalue):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0, shuffle=True);
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X_train, y_train)

    print('Feature Importances')
    print(clf.feature_importances_)
    for feature in zip(X.columns, clf.feature_importances_):
        print(feature)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            clf.estimators_],
                                           axis=0)

    XFeatures = list();


    model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=topN)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test);

    print(model.get_support(indices=True))
    for feature_list_index in model.get_support(indices=True):
        XFeatures.append(X.columns[feature_list_index])

    print('Selected Features')
    print(XFeatures);


    sc = StandardScaler();
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    svclassifier = SVC(kernel="rbf", gamma=1/15, C=cvalue)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test);
    cnfMatrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnfMatrix)
    report = metrics.classification_report(y_test, y_pred)
    print(report)

def predictWithFeatureSelectionNN(X,y,topN,size,learning_rate,n_iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0, shuffle=True);
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X_train, y_train)

    print('Feature Importances')
    print(clf.feature_importances_)
    for feature in zip(X.columns, clf.feature_importances_):
        print(feature)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            clf.estimators_],
                                           axis=0)

    XFeatures = list();

    model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=topN)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test);

    print(model.get_support(indices=True))
    for feature_list_index in model.get_support(indices=True):
        XFeatures.append(X.columns[feature_list_index])

    print('Selected Features')
    print(XFeatures);

    sc = StandardScaler();
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(100, 100, 100),activation='logistic',
                          solver='adam',alpha=0.0001, batch_size='auto',
                          learning_rate='adaptive',learning_rate_init=learning_rate,
                          max_iter=n_iter, shuffle=True, random_state=None, tol=0.0001,
                          verbose=True, warm_start=False, momentum=0.9,
                          nesterovs_momentum=True, early_stopping=True,
                          validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08)

    model.fit(X_train,y_train)
    joblib.dump(model, 'dataset/mlp_class.jbl');

    # Evaluate on training data
    print('\n-- Training data --')
    predictions = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, predictions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(metrics.classification_report(y_train, predictions))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_train, predictions))
    print('')
    stat = list()
    stat.append(round(accuracy * 100.0, 2))
    # Evaluate on test data
    print('\n---- Test data ----')
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(metrics.classification_report(y_test, predictions))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, predictions))
    stat.append(round(accuracy * 100.0, 2))
    stat.append(learning_rate)

    plt.plot(model.loss_curve_)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Model loss for logistic activation and solver adam')
    plt.show()
    return stat;

def predictWithFeatureSelectionNNConst(X,y,topN,size,learning_rate,n_iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0, shuffle=True);
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X_train, y_train)

    print('Feature Importances')
    print(clf.feature_importances_)
    for feature in zip(X.columns, clf.feature_importances_):
        print(feature)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            clf.estimators_],
                                           axis=0)

    XFeatures = list();

    model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=topN)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test);

    print(model.get_support(indices=True))
    for feature_list_index in model.get_support(indices=True):
        XFeatures.append(X.columns[feature_list_index])

    print('Selected Features')
    print(XFeatures);

    sc = StandardScaler();
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                          solver='adam', alpha=0.0001, batch_size='auto',
                          learning_rate='constant', learning_rate_init=learning_rate,
                          max_iter=n_iter, shuffle=True, random_state=None,
                          verbose=True, warm_start=False, momentum=0.9,
                          nesterovs_momentum=True, early_stopping=False,
                          validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, n_iter_no_change=10)

    model.fit(X_train,y_train)
    joblib.dump(model, 'dataset/mlp_class.jbl');
    stat = list()
    # Evaluate on training data
    print('\n-- Training data --')
    predictions = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, predictions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(metrics.classification_report(y_train, predictions))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_train, predictions))
    print('')
    stat.append(round(accuracy * 100.0,2))
    # Evaluate on test data
    print('\n---- Test data ----')
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(metrics.classification_report(y_test, predictions))
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, predictions))
    stat.append(round(accuracy * 100.0, 2))
    stat.append(learning_rate)


    plt.plot(model.loss_curve_)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Model loss for relu activation and solver adam')
    plt.show()
    return stat;


def preprocess(dataset):

    # process left foot
    df = dataset[0][0].sort_values(by=['time']);
    df = df[df['PL1'] < 350]
    df = df[df['PL8'] < 600]
    dataset[0][0] = df

    df = dataset[0][2].sort_values(by=['time']);
    df = df[df['PL6'] < 300]
    dataset[0][2] = df

    df = dataset[0][3].sort_values(by=['time']);
    df = df[df['PL8'] < 300]
    dataset[0][3] = df

    df = dataset[0][4].sort_values(by=['time']);
    df = df[df['PL8'] < 950]
    df = df[df['PL4'] < 180]
    dataset[0][4] = df

    df = dataset[0][5].sort_values(by=['time']);
    df = df[df['PL8'] < 1300]
    dataset[0][5] = df

    df = dataset[1][0].sort_values(by=['time']);
    df = df[df['PR1'] < 300]
    df = df[df['PR2'] < 300]
    df = df[df['PR6'] < 1400]
    df = df[df['PR7'] < 1000]
    dataset[1][0] = df

    df = dataset[1][3].sort_values(by=['time']);
    df = df[df['PR6'] < 800]
    df = df[df['PR7'] < 800]
    df = df[df['PR8'] < 800]
    df = df[df['PR2'] < 300]
    dataset[1][3] = df

    df = dataset[1][5].sort_values(by=['time']);
    df = df[df['PR6'] < 1000]
    df = df[df['PR8'] < 22000]
    dataset[1][5] = df

    return dataset

if __name__ == '__main__':
    # load dataset
    print('Loading Datasets')
    dataset = load_dataset();
    print('Dataset loaded successfully')
    print('Loaded Dataset')
    print(dataset);

    print('Preprocess data')
    dataset = preprocess(dataset);

    print('Plotting Datasets')
    plot_dataset(dataset);
    print('Plotting Completed')

    print('Synchronizing Dataset')
    datalist = synchronize_data(dataset)
    print('Synchronization complete')
    print('Printing Synchronized dataset')
    print(datalist);

    for i in range(len(datalist)):
        activity = datalist[i].activity.iloc[0]
        datalist[i].to_csv(r'dataset/processed/' + activity);

    print('Building Features')
    features = build_features(datalist)
    print('Features built')
    print('Showing Build Features')
    print(features)
    print('First 10 Head of Features')
    print(features.sample(frac=1).head(10))
    features.to_csv(r'dataset/features.csv', index=False)
    print(features.columns)
    y = features['activity'];
    X = features.loc[:, features.columns != 'activity'];

    # Predict using Logistic Regression with test size 0.3
    print('Logistic regression Prediction using test size 0.3')
    predictWithFeatureSelection(X,y,15,0.3)

    print('Logistic regression Prediction using test size 0.2')
    predictWithFeatureSelection(X, y, 15, 0.2)

    # Predict using Logistic Regression with test size 0.3
    print('SVM Prediction using test size 0.3 and regularization(C) value of 15')
    predictWithFeatureSelectionSVM(X, y, 15, 0.3, 15)

    print('SVM Prediction using test size 0.2 and regularization(C) value of 10^6')
    predictWithFeatureSelectionSVM(X, y, 15, 0.2, 10e6)


    dataNN = list()
    dataCon = list()
    print('Neural network classifier for chosen learning rate 0.00625 for activation function relu and solver adam')
    predictWithFeatureSelectionNNConst(X,y,15,0.3,0.00625,1000)

    print('Neural network classifier for chosen learning rate 0.0125 for activation function logistic and solver adam')
    predictWithFeatureSelectionNN(X, y, 15, 0.05, 0.0125, 1000)

    print('Comparing accuracies for multiple learning rate for relu and logistic activation function')
    rate = 0.1
    for i in range(1, 11):
        stat = predictWithFeatureSelectionNNConst(X, y, 15, 0.3, rate, 1000)
        stat1 = predictWithFeatureSelectionNN(X, y, 15, 0.3, rate, 1000)
        dataCon.append(stat);
        dataNN.append(stat1)
        rate = rate / 2.0


    df = pd.DataFrame(dataCon, columns=['TR', 'TE','X'])
    print(df)


    x = np.arange(len(df))
    fig, ax1 = plt.subplots()
    w = 0.3
    plt.xticks(x + w / 2, df['X'])
    train = ax1.bar(x, df['TR'], width=w, color='k', align='center')
    test = ax1.bar(x + w, df['TE'], width=w, color='y', align='center')
    plt.ylabel('Accuracy')
    plt.xlabel('Learning Rate')
    plt.legend([train, test], ['Training', 'Test'])
    plt.title('relu activation and solver adam accuracy')
    plt.show()



    df = pd.DataFrame(dataNN, columns=['TR', 'TE','X'])
    print(df)


    x = np.arange(len(df))
    fig, ax1 = plt.subplots()
    w = 0.3
    plt.xticks(x + w / 2, df['X'])
    train = ax1.bar(x, df['TR'], width=w, color='k', align='center')
    test = ax1.bar(x + w, df['TE'], width=w, color='y', align='center')
    plt.ylabel('Accuracy')
    plt.xlabel('Learning Rate')
    plt.legend([train, test], ['Training', 'Test'])
    plt.title('logistic activation and solver adam accuracy')
    plt.show()



