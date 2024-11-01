import pandas as pd
from sklearn.model_selection import KFold

# 打上intel优化补丁
from sklearnex import patch_sklearn

patch_sklearn()


class SklearnModelStacking:
    def __init__(self, model_params, seed=2024):
        self.model_list = []
        self.split_model_list = []
        self.train_data_f2 = None
        self.test_data_f2 = None
        for item in model_params:
            model = item["model"](random_state=seed, **item["params"])
            self.model_list.append(model)

    def fit(self, X, y):
        # 对外部有可能是分割来的数据集进行重置索引
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # 定义好多个模型的预测结果
        result = []

        kf = KFold(n_splits=5, shuffle=True, random_state=2024)
        for i, model in enumerate(self.model_list):
            temp_model_list = []
            train_data = pd.DataFrame([None] * y.shape[0], columns=[i])
            for train_idx, valid_idx in kf.split(X, y):
                train_x, train_y = X.loc[train_idx], y.loc[train_idx]
                valid_x, valid_y = X.loc[valid_idx], y.loc[valid_idx]
                model.fit(train_x, train_y)
                train_data.loc[valid_idx, i] = model.predict(valid_x)
                temp_model_list.append(model)
            result.append(train_data)
            self.split_model_list.append(temp_model_list)
        self.train_data_f2 = pd.concat(result, axis=1)
        self.train_data_f2['label'] = y

    def predict(self, X_test, stack_model):
        result = []
        for i, models in enumerate(self.split_model_list):
            test_data = pd.Series(0, index=range(X_test.shape[0]))
            for i, model in enumerate(models):
                test_data = test_data + model.predict(X_test)
            test_data = test_data / (i + 1)
            result.append(test_data)
        self.test_data_f2 = pd.concat(result, axis=1)
        return self.true_predict(stack_model)

    def true_predict(self, stack_model):
        X = self.train_data_f2.iloc[:, :-1]
        y = self.train_data_f2.iloc[:, -1]
        X_test = self.test_data_f2
        model = stack_model()

        kf = KFold(n_splits=5, shuffle=True, random_state=2024)
        pre_data = pd.Series(0, index=range(X_test.shape[0]))
        i = 0
        for j, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            train_x, train_y = X.loc[train_idx], y.loc[train_idx]
            model.fit(train_x, train_y)
            pre_data = pre_data + model.predict(X_test)
            i = j
        pre_data = pre_data / (i + 1)
        return pre_data
