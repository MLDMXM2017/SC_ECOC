from sklearn.ensemble import RandomForestClassifier

class feature_selector:
    def __init__(self):
        self.classifier = RandomForestClassifier()

    def fit(self, data, label):
        self.classifier = RandomForestClassifier().fit(data, label)

    def transform(self, data):
        importances = self.classifier.feature_importances_
        importances_mean = 1/len(importances)
        selected_feature = importances > importances_mean
        return data[:, selected_feature]

    def fit_transform(self, data, label):
        self.fit(data, label)
        return self.transform(data)

