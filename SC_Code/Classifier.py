import numpy as np
import copy
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import SC_Code.Distance_Toolkit as Distance_Toolkit
import SC_Code.Matrix_Toolkit as Matrix_Toolkit
import SC_Code.SFFS as SFFS

class BaseECOC:
    def __init__(self, base_estimator=SVC, distance_measure=Distance_Toolkit.euclidean_distance, fill=True, check=True, **estimator_params):
        self.index = []
        self.estimators = []
        self.matrix = None
        self.matrix_without_fill = None
        self.matrix_without_check = None
        self.base_estimator = base_estimator
        self.distance_measure = distance_measure
        self.estimator_params = estimator_params
        self.fill = fill
        self.checking = check

    def create_matrix(self, data, label):
        return np.array([]), []

    def train_col(self, data, label, col):
        train_data, train_label = Matrix_Toolkit.get_data_from_col(data, label, col, self.index)
        estimator = self.base_estimator(**self.estimator_params).fit(train_data, train_label)
        return estimator

    def train_matrix(self, data, label):
        self.matrix, self.index = self.create_matrix(data, label)
        self.matrix = self.check_matrix()
        self.matrix_without_fill = copy.deepcopy(self.matrix)
        if self.fill or self.checking:
            self.fillzero(data, label)
        self.estimators = []
        for i in range(self.matrix.shape[1]):
            self.estimators.append(self.train_col(data, label, self.matrix[:, i]))
        return self.matrix, self.estimators

    def fit(self, data, label):
        self.train_matrix(data, label)

    def predict(self, data):
        if not self.estimators:
            raise ValueError('This Model has not been trained!')
        vectors = self.predict_vector(data)
        labels = self.vectors_to_labels(vectors)
        return labels

    def predict_vector(self, data):
        vectors = None
        for estimator in self.estimators:
            if vectors is None:
                vectors = estimator.predict(data).reshape((-1, 1))
            else:
                vectors = np.hstack((vectors, estimator.predict(data).reshape((-1, 1))))
        return vectors

    def vectors_to_labels(self, vectors, weights=None):
        labels = []
        for vector in vectors:
            distance = np.inf
            label = self.index[0]
            for matrix_index in range(len(self.matrix)):
                matrix_row = self.matrix[matrix_index, :]
                distance_temp = self.distance_measure(vector, matrix_row, weights)
                if distance_temp < distance:
                    distance = distance_temp
                    label = self.index[matrix_index]
            labels.append(label)
        return labels

    def check_matrix(self):
        # self.matrix, index = np.unique(self.matrix, axis=1, return_index=True)
        ## self.estimators = [self.estimators[i] for i in index]
        index_to_delete = []
        for i in range(self.matrix.shape[1]):
            if not Matrix_Toolkit.exist_two_class_for_col(self.matrix[:, i]):
                index_to_delete.append(i)
        self.matrix = np.delete(self.matrix, index_to_delete, axis=1)
        # self.estimators = [self.estimators[i] for i in range(len(self.estimators)) if i not in index_to_delete]
        return self.matrix

    def check_column(self, column):
        column = column.reshape([1,-1])
        sign_column = np.sign(column)
        if (1 not in sign_column) or (-1 not in sign_column):
            return False
        if self.matrix is None:
            return True
        sign_matrix = np.sign(self.matrix)
        for i in range(sign_matrix.shape[1]):
            if np.all([sign_column == sign_matrix[:, i]]) or \
                    np.all([sign_column == sign_matrix[:, i]]):
                return False
        return True

    def fillzero(self, data, label):
        pass


class DataBasedECOC(BaseECOC):
    def __init__(self, base_estimator=SVR, distance_measure=Distance_Toolkit.euclidean_distance,
                 coverage='normal', fill=True, check=True, **estimator_params):
        BaseECOC.__init__(self, base_estimator, distance_measure, fill, check, **estimator_params)
        self.coverage=coverage
        if coverage == 'normal':
            self.cover_rate = Matrix_Toolkit.coverage_rate
        elif coverage == 'distance':
            self.cover_rate = Matrix_Toolkit.distance_coverage_rate
    def check(self, data, label):
        check_matrix = np.zeros(self.matrix.shape)
        weights = np.zeros((1, self.matrix.shape[1]))
        for i in range(self.matrix.shape[1]):
            col = self.matrix[:, i]
            new_label = [col[n] for lab in label for n in range(len(self.index)) if lab == self.index[n]]
            pred_label = self.estimators[i].predict(data)
            mse = mean_squared_error(new_label, pred_label)
            weights[0, i] = np.exp(-mse)
            for k in range(len(self.index)):
                data_temp = np.array([data[j, :] for j in range(data.shape[0]) if label[j] == self.index[k]])
                check_matrix[k, i] = np.mean(self.estimators[i].predict(data_temp))
        # print('before:\n',self.matrix)
        # self.matrix = (self.matrix + check_matrix * weights) / (np.ones(self.matrix.shape) + weights)
        self.matrix = (self.matrix + check_matrix * (weights)) / (np.ones(self.matrix.shape) + (weights))
        # print('after:\n',self.matrix)

    def fit(self, data, label):
        data, label = Matrix_Toolkit.duplicate_single_class(data, label)
        self.train_matrix(data, label)
        self.matrix_without_check = copy.deepcopy(self.matrix)
        if self.checking:
            self.check(data, label)

    def drop_bad_col(self, validation_data, validation_label):
        index_to_move = []
        for i, estimator in enumerate(self.estimators):
            data_temp, label_temp = Matrix_Toolkit.get_data_from_col(validation_data, validation_label,
                                                                     self.matrix[:, i], self.index, hardcode=True)
            pred_label = np.sign(estimator.predict(data_temp))
            if accuracy_score(label_temp, pred_label) < 0.5:
                index_to_move.append(i)
        self.matrix = np.delete(self.matrix, index_to_move, axis=1)
        self.estimators = [self.estimators[i] for i in range(len(self.estimators))
                           if i not in index_to_move]

    def fill_column_zero(self, data, label, column):
        column = copy.deepcopy(column)
        pos_to_fill = [index for index in range(len(column)) if column[index]==0]
        positive_label = [self.index[index] for index in range(len(column)) if column[index] > 0]
        negative_label = [self.index[index] for index in range(len(column)) if column[index] < 0]
        positive_data, positive_label = Matrix_Toolkit.get_data_by_label(data, label, positive_label)
        negative_data, negative_label = Matrix_Toolkit.get_data_by_label(data, label, negative_label)
        positive_center = np.mean(positive_data, axis=0)
        negative_center = np.mean(negative_data, axis=0)
        for i in pos_to_fill:
            target_label = [self.index[i]]
            data_temp, label_temp =Matrix_Toolkit.get_data_by_label(data, label, target_label)
            if self.coverage == 'normal':
                group = [self.distance_measure(data, positive_center) < self.distance_measure(data, negative_center) for data in data_temp]
                positive_num = group.count(True)
                negative_num = group.count(False)
                if positive_num >negative_num:
                    column[i] = positive_num / (positive_num + negative_num)
                else:
                    column[i] = -negative_num / (positive_num + negative_num)
            elif self.coverage == 'distance':
                positive_distance_mean = np.mean([self.distance_measure(data, positive_center) for data in data_temp])
                negative_distance_mean = np.mean([self.distance_measure(data, negative_center) for data in data_temp])
                if positive_distance_mean < negative_distance_mean:
                    column[i] = 2/(1 + np.exp(-positive_distance_mean)) - 1
                else:
                    column[i] = 2/(1 + np.exp(negative_distance_mean)) - 1
        return column
    # def fill_column_zero(self, data, label, column, column_num):
    #     column = copy.deepcopy(column)
    #     pos_to_fill = [index for index in range(len(column)) if column[index]==0]
    #     for i in pos_to_fill:
    #         target_label = [self.index[i]]
    #         data_temp, label_temp =Matrix_Toolkit.get_data_by_label(data, label, target_label)
    #         column[i] = np.mean(self.estimators[column_num].predict(data_temp))
    #     return column
    def fillzero(self, data, label):
        for i in range(self.matrix.shape[1]):
            self.matrix[:, i] = self.fill_column_zero(data, label, self.matrix[:, i])
        return self.matrix

    def train_col(self, data, label, col):
        train_data, train_label = Matrix_Toolkit.get_data_from_col(data, label, col, self.index, hardcode=False)
        estimator = self.base_estimator(**self.estimator_params).fit(train_data, train_label)
        return estimator


class SC_ECOC(DataBasedECOC):
    def predict(self, data):
        if not self.estimators:
            raise ValueError('This Model has not been trained!')
        vectors = self.predict_vector(data)
        labels = self.vectors_to_labels(vectors)
        return labels, vectors
    def create_matrix(self, data, label):
        index = list(np.unique(label))
        matrix = None
        label_to_divide = [index]
        while len(label_to_divide)>0:
            label_set = label_to_divide.pop(0)
            data_temp, label_temp = Matrix_Toolkit.get_data_by_label(data, label, label_set)
            positive_label_set, negative_label_set = SFFS.sffs_divide(data_temp, label_temp)
            positive_data, positive_label = Matrix_Toolkit.get_data_by_label(data, label, positive_label_set)
            negative_data, negative_label = Matrix_Toolkit.get_data_by_label(data, label, negative_label_set)
            rate = self.cover_rate(positive_data, positive_label, negative_data, negative_label)
            new_column = np.zeros((len(index), 1))
            for key in rate:
                new_column[index.index(key)] = rate[key]
            if matrix is None:
                matrix = new_column
            else:
                matrix = np.hstack((matrix, new_column))
            if len(positive_label_set) > 1:
                label_to_divide.insert(0, positive_label_set)
            if len(negative_label_set) > 1:
                label_to_divide.insert(0, negative_label_set)
        return matrix, index

    def add_columns(self, data, label):
        # matrix_sum = np.sum(self.matrix)
        # matrix_mean = matrix_sum / np.sum(self.matrix != 0)
        # row_mean = np.mean(self.matrix, axis=1)
        # mask = row_mean > matrix_mean
        # while len(mask) > 0:
        while True:
            if end_flag:
                break
            matrix_mean = np.mean(self.matrix[self.matrix.nonzero()])
            row_mean = []
            for i in range(self.matrix.shape[0]):
                row = self.matrix[i, :]
                row_mean.append(np.mean(row[row.nonzero()]))
            row_mean = np.array(row_mean)
            row_index = np.array(self.index)[row_mean < matrix_mean]
            label_to_enforce = [self.index[i] for i in range(len(self.index)) if self.index[i] in row_index]
            if len(label_to_enforce) < 2:
                break
            while True:
                train_data_temp, train_label_temp = Matrix_Toolkit.get_data_by_label(data, label,
                                                                                     label_to_enforce)
                positive_label, negative_label = SFFS.sffs_divide(train_data_temp, train_label_temp)
                data_positive, label_positive = Matrix_Toolkit.get_data_by_label(train_data_temp, train_label_temp,
                                                                                 positive_label)
                data_negative, label_negative = Matrix_Toolkit.get_data_by_label(train_data_temp, train_label_temp,
                                                                                 negative_label)
                rate = self.cover_rate(data_positive, label_positive, data_negative, label_negative,
                                       distance_measure=self.distance_measure)
                new_column = np.zeros((self.matrix.shape[0], 1))
                for label in positive_label:
                    new_column[self.index.index(label)] = rate[label]
                for label in negative_label:
                    new_column[self.index.index(label)] = rate[label]
                if not self.check_column(new_column):
                    label_to_enforce.pop()
                    if len(label_to_enforce) < 2:
                        end_flag = True
                        break
                else:
                    new_estimator = self.train_col(data, label, new_column)
                    self.matrix = np.hstack((self.matrix, new_column))
                    self.estimators.append(new_estimator)
                    break



