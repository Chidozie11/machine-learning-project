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

def load_dataset_acc(path, y):
    data = list();
    path = path.strip('/');
    acc_directory = path
    for name in listdir(acc_directory):
        filename = acc_directory + '/' + name;
        df = pd.read_csv(filename, header=0, names=['time', 'AX', 'AY', 'AZ']);
        df.dropna(axis=0, how='any', inplace=True)
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
        df.dropna(axis=0, how='any', inplace=True)
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
        df.dropna(axis=0, how='any', inplace=True)
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
        df.dropna(axis=0, how='any', inplace=True)
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
        plt.title(data.activity[0])
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
        plt.title(data.activity[0])
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
    dataset_pressureleft.append(load_dataset_pressure_left('dataset/running/left', 'running'));
    dataset_pressureleft.append(load_dataset_pressure_left('dataset/sitting/left', 'sitting'));
    dataset_pressureleft.append(load_dataset_pressure_left('dataset/standing/left', 'standing'));
    dataset_pressureleft.append(load_dataset_pressure_left('dataset/walking/left', 'walking'));
    dataset_pressureleft.append(load_dataset_pressure_left('dataset/astair/left', 'ascendingstair'));
    dataset_pressureleft.append(load_dataset_pressure_left('dataset/dstair/left', 'descendingstair'));

    dataset_pressure_right = list();
    dataset_pressure_right.append(load_dataset_pressure_right('dataset/running/right', 'running'));
    dataset_pressure_right.append(load_dataset_pressure_right('dataset/sitting/right', 'sitting'));
    dataset_pressure_right.append(load_dataset_pressure_right('dataset/standing/right', 'standing'));
    dataset_pressure_right.append(load_dataset_pressure_right('dataset/walking/right', 'walking'));
    dataset_pressure_right.append(load_dataset_pressure_right('dataset/astair/right', 'ascendingstair'));
    dataset_pressure_right.append(load_dataset_pressure_right('dataset/dstair/right', 'descendingstair'));

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
        activity = datalist['activity'][0]
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
        datalist = datalist.fillna(0);
        datalist['PL'] = datalist[['PL1', 'PL2', 'PL3', 'PL4', 'PL5', 'PL6', 'PL7', 'PL8']].replace(0, np.NaN).mean(
            axis=1, skipna=True);
        datalist['AR'] = np.sqrt(np.power(datalist['AX'],2) + np.power(datalist['AY'],2) + np.power(datalist['AZ'], 2));
        datalist['GR'] = np.sqrt(np.power(datalist['GX'],2) + np.power(datalist['GY'],2) + np.power(datalist['GZ'], 2));
        datalist['PR'] = datalist[['PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'PR7', 'PR8']].replace(0, np.NaN).mean(
            axis=1, skipna=True);
        datalist = datalist.dropna().drop(columns=['GX','GY','GZ','AX','AY','AZ','PL1', 'PL2', 'PL3', 'PL4', 'PL5', 'PL6', 'PL7', 'PL8','PR1', 'PR2', 'PR3', 'PR4', 'PR5', 'PR6', 'PR7', 'PR8'], axis=1)
        featureslist.append(datalist);
    return featureslist;

def build_features(featurelist):
    columns = ['PLMEAN','PLMEDIAN','PLSTD','PLVAR','PLSKEW',
               'PRMEAN','PRMEDIAN','PRSTD','PRVAR','PRSKEW',
               'AMEAN','AMEDIAN','ASTD','AVAR','ASKEW',
               'GMEAN','GMEDIAN','GSTD','GVAR','GSKEW',
               'activity'];
    df = pd.DataFrame(columns=columns);
    datafinal = list();
    for i in range(len(featurelist)):
        data = featurelist[i];
        activity = data['activity'].iloc[0]
        data_length = len(data);
        mylist = list();
        for beg in range(0,data_length,2):
            mylist.append(list(itertools.chain.from_iterable(data.iloc[beg:(beg + 2 if beg + 2 < data_length else data_length)][['PL']].values)));
            mylist.append(list(itertools.chain.from_iterable(data.iloc[beg:(beg + 3 if beg + 3 < data_length else data_length)][['PR']].values)));
            mylist.append(list(itertools.chain.from_iterable(data.iloc[beg:(beg + 3 if beg + 3 < data_length else data_length)][['AR']].values)));
            mylist.append(list(itertools.chain.from_iterable(data.iloc[beg:(beg + 3 if beg + 3 < data_length else data_length)][['GR']].values)));

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                row_df = pd.DataFrame({
                columns[0]: [np.nanmean(mylist[0])],
                columns[1]: [np.nanmedian(mylist[0])],
                columns[2]: [np.nanstd(mylist[0])],
                columns[3]: [np.nanvar(mylist[0])],
                columns[4]: [sp.stats.skew(mylist[0])],
                columns[5]: [np.nanmean(mylist[1])],
                columns[6]: [np.nanmedian(mylist[1])],
                columns[7]: [np.nanstd(mylist[1])],
                columns[8]: [np.nanvar(mylist[1])],
                columns[9]: [sp.stats.skew(mylist[1])],
                columns[10]: [np.nanmean(mylist[2])],
                columns[11]: [np.nanmedian(mylist[2])],
                columns[12]: [np.nanstd(mylist[2])],
                columns[13]: [np.nanvar(mylist[2])],
                columns[14]: [sp.stats.skew(mylist[2])],
                columns[15]: [np.nanmean(mylist[3])],
                columns[16]: [np.nanmedian(mylist[3])],
                columns[17]: [np.nanstd(mylist[3])],
                columns[18]: [np.nanvar(mylist[3])],
                columns[19]: [sp.stats.skew(mylist[3])],
                columns[20]: [activity]
                })
            if not(row_df.empty):
                df = df.append(row_df, ignore_index=True)
        print('Feature building: ', (i * 100 // 6), '% complete');
    return df;

def select_features_with_rank(X,y,topN):
    #clf = logi(n_estimators=50)
    clf = clf.fit(X, y)
    print('Feature Importances')
    print(clf.feature_importances_)
    for feature in zip(X.columns, clf.feature_importances_):
        print(feature)
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            clf.estimators_],
                                           axis=0)

    model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features= topN)
    X_new = model.transform(X)
    featList = list();
    XFeatures = list();

    for feature_list_index in model.get_support(indices=True):
        XFeatures.append(X.columns[feature_list_index])
    featList.append(XFeatures);
    featList.append(y);

    # plt.bar(X.columns, feature_importance_normalized, width=0.8)
    # plt.xlabel('Feature Labels')
    # plt.ylabel('Feature Importances')
    # plt.title('Comparison of different Feature Importances')
    # plt.xticks(rotation=90)
    # plt.show()
    return featList;


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

if __name__ == '__main__':
    # load dataset
    print('Loading Datasets')
    dataset = load_dataset();
    print('Dataset loaded successfully')
    print('Loaded Dataset')
    #print(dataset);

    print('Plotting Datasets')
    #plot_dataset(dataset);
    print('Plotting Completed')

    print('Synchronizing Dataset')
    datalist = synchronize_data(dataset)
    print('Synchronization complete')
    print('Printing Synchronized dataset')
    #print(datalist);

    print('Building Features')
    features = build_features(datalist)
    print('Features built')
    print('Showing Build Features')
    print(features)
    features.to_csv(r'dataset/features.csv', index=False)

    y = features['activity'];
    X = features.loc[:, features.columns != 'activity'];
    #predictWithoutFeatureSelection(X,y)
    #print('Prediction accuracy using all features', pred1)
    predictWithFeatureSelection(X,y, 0.2, 15)
    predictWithFeatureSelection(X,y, 0.3, 15)
    predictWithFeatureSelection(X,y, 0.3, 17)
    predictWithFeatureSelection(X,y, 0.2, 18)
    
    #print('Prediction accuracy using selected features', pred2)


    print('selecting relevant features based on rank and importance.')
    print('Note: relevant features are created based on summary of built features')
    #selected_features = select_features_with_rank(X_train,y_train,15);

    


    #print(selected_features[0])

