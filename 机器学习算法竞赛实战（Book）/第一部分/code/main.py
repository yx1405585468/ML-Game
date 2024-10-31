from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from StackingRegressor_model import SklearnModelStacking
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # 导入加利福尼亚房价数据集
    california_housing = fetch_california_housing()

    # 创建 DataFrame
    data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)

    # 添加目标列
    data['target'] = california_housing.target
    model_params_ = [
        {
            "model": RandomForestRegressor,
            "params": {'n_estimators': 100, 'max_features': 0.2, 'max_depth': 12, 'min_samples_leaf': 2}
        },
        {
            "model": ExtraTreesRegressor,
            "params": {'n_estimators': 100, 'max_features': 0.2, 'max_depth': 12, 'min_samples_leaf': 2}
        }
    ]

    model = SklearnModelStacking(model_params_)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 构造训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model.fit(X_train, y_train)
    pre = model.predict(X_test, RandomForestRegressor)
    print(mean_squared_error(y_test, pre))

    param = {'n_estimators': 100, 'max_features': 0.2, 'max_depth': 12, 'min_samples_leaf': 2}
    model = RandomForestRegressor(**param)
    model.fit(X_train, y_train)
    pre = model.predict(X_test)
    print(mean_squared_error(y_test, pre))
