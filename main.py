from SC_Code.Classifier import SC_ECOC
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from SC_Code.data_preprocess import custom_preprocess
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import platform
import sys
import time
import json
import warnings

warnings.filterwarnings('ignore')

def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.power(np.array(x)-np.array(y), 2), axis=0))

def hamming_distance(x, y):
    return np.sum(x!=y)

def code_difference(matrix, distance=euclidean_distance):
    distance_matrix = np.zeros([len(matrix), len(matrix)])
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            mask_x = np.array(matrix[i]) != 0
            mask_y = np.array(matrix[j]) != 0
            mask = np.logical_and(mask_x, mask_y)
            x = matrix[i][mask]
            y = matrix[j][mask]
            d = distance(x, y)
            distance_matrix[i][j] = d
            distance_matrix[j][i] = d
        np.round(distance_matrix, 2)
    return distance_matrix

def plot_data(classifier, test_label, predict_label, path=r'/Users/kaijiefeng/Desktop/'):
    appendix = '.png'
    # if classifier.fill_zeros:
    #     appendix = '_filled'+appendix
    # else:
    #     appendix = '_not_filled'+appendix
    accuracy = accuracy_score(test_label, predict_label)
    # print(file + ' Accuracy:', accuracy)
    # print('class number:', len(classifier.index))
    # print('matrix shape:', classifier.matrix.shape)
    # print()
    err = list(range(len(classifier.matrix[0])))
    plt.figure(figsize=[classifier.matrix.shape[1], classifier.matrix.shape[0]])
    sns.heatmap(classifier.matrix, annot=True)
    # plt.axis('off')
    x_pos = [i + 0.5 for i in range(len(err))]
    plt.xticks(x_pos, err)
    print(classifier.index)
    y_pos = [i + 0.5 for i in range(len(classifier.index))]
    plt.yticks(y_pos, classifier.index, rotation=0)
    plt.savefig(path + file[:file.find(r'.')] + str(round(accuracy, 2)) + appendix)
    # print(path + file[:file.find(r'.')] + str(round(accuracy, 2)) + appendix)
    plt.close()

    plt.figure()
    sns.heatmap(code_difference(classifier.matrix), annot=True)
    plt.savefig(path + file[:file.find(r'.')] + str(round(accuracy, 2)) + 'difference' + appendix)
    plt.close()

    plt.figure()
    sns.heatmap(confusion_matrix(test_label, predict_label), annot=True)
    plt.savefig(path + file[:file.find(r'.')] + str(round(accuracy, 2)) + 'conf_matrix' + appendix)
    plt.close()

def read_data(path):
    df = pd.read_csv(path, header=None, index_col=None)
    data = df.values[:,:-1]
    label = df.values[:,-1]
    data = custom_preprocess(StandardScaler).fit_transform(data)
    return data, label

def get_branch(matrix, index):
    matrix = np.array(matrix)
    if len(matrix)==0:
        return []
    res = []
    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        branch = {'pos':[],'neg':[]}
        for row, value in enumerate(column):
            if value > 0:
                branch['pos'].append(str(index[row]))
            elif value < 0:
                branch['neg'].append(str(index[row]))
        res.append(branch)
    return res

def runner(classifier, classifier_name, train_data, train_label, test_data, test_label, root_path, data_name, iteration):
    save_path = os.path.join(root_path, classifier_name, data_name, iteration)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start_time = time.time()
    classifier.fit(train_data, train_label)
    train_time = time.time() - start_time
    predict_label = []
    predict_code = []
    start_time = time.time()
    if classifier_name.startswith('SC_ECOC'):
        predict_label, predict_code = classifier.predict(test_data)
    else:
        predict_label = classifier.predict(test_data)
    predict_time = time.time() - start_time
    #存储预测结果
    df = pd.DataFrame(data=test_data, index=None, columns=['feature_'+str(i) for i in range(test_data.shape[1])])
    df['true_label'] = test_label
    df['predict_label'] = predict_label
    prediction_save_path = os.path.join(save_path, 'prediction_records.csv')
    df.to_csv(prediction_save_path, header=True, index=True)
    if len(predict_code) > 0:
        predict_code_save_path = os.path.join(save_path, 'predict_code.csv')
        df = pd.DataFrame(predict_code, index=None, columns=['code_'+str(i) for i in range(predict_code.shape[1])])
        df.to_csv(predict_code_save_path, header=True, index=True)
    #存储训练数据
    if classifier_name.startswith('SC_ECOC'):
        predict_label, predict_code = classifier.predict(train_data)
    else:
        predict_label = classifier.predict(train_data)
    predict_time = time.time() - start_time
    df = pd.DataFrame(data=train_data, index=None, columns=['feature_'+str(i) for i in range(test_data.shape[1])])
    df['true_label'] = train_label
    df['predict_label'] = predict_label
    prediction_save_path = os.path.join(save_path, 'prediction_train_records.csv')
    df.to_csv(prediction_save_path, header=True, index=True)
    if len(predict_code) > 0:
        predict_code_save_path = os.path.join(save_path, 'predict_train_code.csv')
        df = pd.DataFrame(predict_code, index=None, columns=['code_'+str(i) for i in range(predict_code.shape[1])])
        df.to_csv(predict_code_save_path, header=True, index=True)
    #存储index
    index_save_path = os.path.join(save_path, 'index.json')
    with open(index_save_path, 'w') as f:
        if isinstance(classifier.index, dict):
            temp = [str(i) for i in list(classifier.index.keys())]
            json.dump(temp, f)
        else:
            temp = [str(i) for i in classifier.index]
            json.dump(temp, f)
    #存储编码矩阵
    matrix_save_path = os.path.join(save_path, 'matrix.json')
    with open(matrix_save_path, 'w') as f:
        json.dump(classifier.matrix.tolist(), f)
    #对SC_ECOC相关的矩阵存储未填充0和未check的编码矩阵
    if classifier_name.startswith('SC_ECOC'):
        matrix_without_fill_save_path = os.path.join(save_path, 'matrix_without_fill.json')
        with open(matrix_without_fill_save_path, 'w') as f:
            json.dump(classifier.matrix_without_fill.tolist(), f)
        matrix_without_check_save_path = os.path.join(save_path, 'matrix_without_check.json')
        with open(matrix_without_check_save_path, 'w') as f:
            json.dump(classifier.matrix_without_check.tolist(), f)
        #存储矩阵分支结构
        branch = get_branch(classifier.matrix_without_fill, classifier.index)
        branch_save_path = os.path.join(save_path, 'branch.json')
        with open(branch_save_path, 'w') as f:
            json.dump(branch, f)
    #存储运行时间
    time_save_path = os.path.join(save_path, 'time.json')
    with open(time_save_path, 'w') as f:
        json.dump({'train_time':train_time, 'predict_time':predict_time}, f)

def plot_matrix(matrix, index, save_path, title):
    matrix = np.round(matrix, 2)
    plt.figure(figsize=[matrix.shape[1], matrix.shape[0]], dpi=300)
    sns.heatmap(matrix, annot=True, cbar=True, yticklabels=index)
    plt.yticks(rotation=360)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_bar(height, index, save_path, title):
    height = np.round(height, 2)
    plt.figure(figsize=[len(height)*1.6,len(height)])
    plt.bar(list(range(1,len(height)+1)), height, tick_label=index)
    plt.title(title)
    plt.ylim(0, 120)
    plt.ylabel('Recall score')
    plt.xlabel('Labels')
    for i, v in enumerate(height):
        plt.text(i+1, v+5, str(v)+'%', ha='center', fontsize=10)
    plt.savefig(save_path)
    plt.close()


def metric_fusion(root_path):
    method_dicts = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    method_list = [os.path.split(d)[1] for d in method_dicts]
    for method_dict in method_dicts:
        data_dicts = [os.path.join(method_dict, d) for d in os.listdir(method_dict) if os.path.isdir(os.path.join(method_dict, d))]
        data_list = [os.path.split(d)[1] for d in data_dicts]
        all_metrics = {}
        for data_dict in data_dicts:
            iter_dicts = [os.path.join(data_dict, d) for d in os.listdir(data_dict) if os.path.isdir(os.path.join(data_dict, d))]
            metrics = {}
            for iter_dict in iter_dicts:
                with open(os.path.join(iter_dict, 'index.json'), 'r') as f:
                    index = json.load(f)
                    if type(index)=='dict':
                        index = list(index.keys())
                with open(os.path.join(iter_dict, 'matrix.json'), 'r') as f:
                    matrix = json.load(f)
                    title = "The matrix of " + os.path.split(data_dict)[1] + " with " + os.path.split(method_dict)[1]
                    save_path = os.path.join(iter_dict, 'matrix.png')
                    plot_matrix(matrix, index, save_path, title)
                if os.path.exists(os.path.join(iter_dict, 'matrix_without_fill.json')):
                    with open(os.path.join(iter_dict, 'matrix_without_fill.json'), 'r') as f:
                        matrix = json.load(f)
                        title = "The matrix without fill of " + os.path.split(data_dict)[1] + " with " + os.path.split(method_dict)[1]
                        save_path = os.path.join(iter_dict, 'matrix_without_fill.png')
                        plot_matrix(matrix, index, save_path, title)
                if os.path.exists(os.path.join(iter_dict, 'matrix_without_check.json')):
                    with open(os.path.join(iter_dict, 'matrix_without_check.json'), 'r') as f:
                        matrix = json.load(f)
                        title = "The matrix without check of " + os.path.split(data_dict)[1] + " with " + os.path.split(method_dict)[1]
                        save_path = os.path.join(iter_dict, 'matrix_without_check.png')
                        plot_matrix(matrix, index, save_path, title)
                df_pr = pd.read_csv(os.path.join(iter_dict, 'prediction_records.csv'), header=0, index_col=0)
                true_label = df_pr['true_label'].tolist()
                predict_label = df_pr['predict_label'].tolist()
                #计算和存储confusion matrix
                unique_label = np.unique(true_label + predict_label)
                conf_matrix = confusion_matrix(true_label, predict_label)
                title = "Conf matrix of " + os.path.split(data_dict)[1] + " with " + os.path.split(method_dict)[1]
                save_path = os.path.join(iter_dict, 'confusion_matrix.png')
                plot_matrix(conf_matrix, unique_label, save_path, title)
                recall = recall_score(true_label, predict_label, average=None) * 100
                title = "Recall score of" + os.path.split(data_dict)[1] + " with " + os.path.split(method_dict)[1]
                save_path = os.path.join(iter_dict, 'recall_score.png')
                plot_bar(recall, unique_label, save_path, title)
                accuracy = accuracy_score(true_label, predict_label)
                metrics['accuracy'] = metrics.get('accuracy', []) + [accuracy]
                fscore = f1_score(true_label, predict_label, average='macro')
                metrics['fscore'] = metrics.get('fscore', []) + [fscore]
                mcc = matthews_corrcoef(true_label, predict_label)
                metrics['mcc'] = metrics.get('mcc', []) + [mcc]
            df = pd.DataFrame(metrics['accuracy'], index=None, columns=None)
            save_path = os.path.join(data_dict, '{}_{}_{}.csv'.format('Accuracy',os.path.split(data_dict)[1], os.path.split(method_dict)[1]))
            df.to_csv(save_path, header=True, index=True)
            all_metrics['accuracy'] = all_metrics.get('accuracy', []) + [metrics['accuracy']]
            df = pd.DataFrame(metrics['fscore'], index=None, columns=None)
            save_path = os.path.join(data_dict, '{}_{}_{}.csv'.format('Fscore',os.path.split(data_dict)[1], os.path.split(method_dict)[1]))
            df.to_csv(save_path, header=True, index=True)
            all_metrics['fscore'] = all_metrics.get('fscore', []) + [metrics['fscore']]
            df = pd.DataFrame(metrics['mcc'], index=None, columns=None)
            save_path = os.path.join(data_dict, '{}_{}_{}.csv'.format('MCC',os.path.split(data_dict)[1], os.path.split(method_dict)[1]))
            df.to_csv(save_path, header=True, index=True)
            all_metrics['mcc'] = all_metrics.get('mcc', []) + [metrics['mcc']]
        df = pd.DataFrame(np.array(all_metrics['accuracy']).T, columns=data_list, index=None)
        save_path = os.path.join(method_dict, '{}_{}.csv'.format('Accuracy', os.path.split(method_dict)[1]))
        df.to_csv(save_path, header=True, index=True)
        df = pd.DataFrame(np.array(all_metrics['fscore']).T, columns=data_list, index=None)
        save_path = os.path.join(method_dict, '{}_{}.csv'.format('Fscore', os.path.split(method_dict)[1]))
        df.to_csv(save_path, header=True, index=True)
        df = pd.DataFrame(np.array(all_metrics['mcc']).T, columns=data_list, index=None)
        save_path = os.path.join(method_dict, '{}_{}.csv'.format('MCC', os.path.split(method_dict)[1]))
        df.to_csv(save_path, header=True, index=True)
    #将所有结果整合在一起
    accuracy_file = [os.path.join(d,'{}_{}.csv'.format('Accuracy',os.path.split(d)[1])) for d in method_dicts]
    fscore_file = [os.path.join(d,'{}_{}.csv'.format('Fscore',os.path.split(d)[1])) for d in method_dicts]
    mcc_file = [os.path.join(d,'{}_{}.csv'.format('MCC',os.path.split(d)[1])) for d in method_dicts]

    def top_k(x, k=10):
        return x[np.argsort(x)[-k:]]

    accuracy_mean_result = []
    accuracy_std_result = []
    for f in accuracy_file:
        df = pd.read_csv(f, header=0, index_col=0)
        if f.find('SC_ECOC')!=-1:
            columns_temp = df.columns
            df = df.values
            df = np.apply_along_axis(lambda x:top_k(x), 0, df)
            df = pd.DataFrame(df, columns=columns_temp, index=None)
        mean = np.mean(df.values, axis=0)
        std = np.std(df.values, axis=0)
        accuracy_mean_result.append(mean.tolist())
        accuracy_std_result.append(std.tolist())
    save_path = os.path.join(root_path, 'Accuracy_mean_result.csv')
    df = pd.DataFrame(np.array(accuracy_mean_result).T, index=data_list, columns=method_list)
    df.to_csv(save_path, header=True, index=True)
    save_path = os.path.join(root_path, 'Accuracy_std_result.csv')
    df = pd.DataFrame(np.array(accuracy_std_result).T, index=data_list, columns=method_list)
    df.to_csv(save_path, index=data_list, columns=method_list)

    fscore_mean_result = []
    fscore_std_result = []
    for f in fscore_file:
        df = pd.read_csv(f, header=0, index_col=0)
        if f.find('SC_ECOC')!=-1:
            columns_temp = df.columns
            df = df.values
            df = np.apply_along_axis(lambda x:top_k(x), 0, df)
            df = pd.DataFrame(df, columns=columns_temp, index=None)
        mean = np.mean(df.values, axis=0)
        std = np.std(df.values, axis=0)
        fscore_mean_result.append(mean.tolist())
        fscore_std_result.append(std.tolist())
    save_path = os.path.join(root_path, 'Fscore_mean_result.csv')
    df = pd.DataFrame(np.array(fscore_mean_result).T, index=data_list, columns=method_list)
    df.to_csv(save_path, header=True, index=True)
    save_path = os.path.join(root_path, 'Fscore_std_result.csv')
    df = pd.DataFrame(np.array(fscore_std_result).T, index=data_list, columns=method_list)
    df.to_csv(save_path, index=data_list, columns=method_list)

    mcc_mean_result = []
    mcc_std_result = []
    for f in mcc_file:
        df = pd.read_csv(f, header=0, index_col=0)
        if f.find('SC_ECOC')!=-1:
            columns_temp = df.columns
            df = df.values
            df = np.apply_along_axis(lambda x:top_k(x), 0, df)
            df = pd.DataFrame(df, columns=columns_temp, index=None)
        mean = np.mean(df.values, axis=0)
        std = np.std(df.values, axis=0)
        mcc_mean_result.append(mean.tolist())
        mcc_std_result.append(std.tolist())
    save_path = os.path.join(root_path, 'MCC_mean_result.csv')
    df = pd.DataFrame(np.array(mcc_mean_result).T, index=data_list, columns=method_list)
    df.to_csv(save_path, header=True, index=True)
    save_path = os.path.join(root_path, 'MCC_std_result.csv')
    df = pd.DataFrame(np.array(mcc_std_result).T, index=data_list, columns=method_list)
    df.to_csv(save_path, index=data_list, columns=method_list)









if __name__ == '__main__':
    start_time = time.time()
    r = SVR
    classifiers = [SC_ECOC(base_estimator=r)]
    classifiers_name = ['SC_ECOC']
    data_path = r'./Data/'
    file_list = ['dermatology.csv']
    folds = 2
    pool_size = 1
    # save result path
    root_path = r'./sc_ecoc_result/'
    print(file_list)
    pool = Pool(processes=pool_size)
    for file in file_list:
        file_path = os.path.join(data_path, file)
        data, label = read_data(file_path)
        for classifier_index, classifier in enumerate(classifiers):
            for i in range(folds):
                from copy import deepcopy
                data_temp = deepcopy(data)
                label_temp = deepcopy(label)
                train_data, test_data, train_label, test_label = train_test_split(data_temp, label_temp, test_size=0.2)
                param = {}
                # runner is startup class
                pool.apply_async(runner, kwds={'classifier':classifier, 'classifier_name': classifiers_name[classifier_index],
                                         'train_data':train_data, 'train_label':train_label,
                                         'test_data':test_data, 'test_label':test_label,
                                         'root_path': root_path,
                                         'data_name': file[:file.find('.')], 'iteration': str(i)})
    pool.close()
    pool.join()
    pool.terminate()
    metric_fusion(root_path)

    print('using time:', time.time()-start_time,' secs')
