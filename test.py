import bentoml



iris_clf_runner= bentoml.sklearn.get("iris_clf:latest").to_runner
iris_clf_runner.init_local()

print(iris_clf_runner.predict([[5.1, 3.5, 1.4, 0.2]]))