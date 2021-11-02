import numpy as np
from macest.classification import models as cl_mod
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

X,y = datasets.make_circles(n_samples= 2 * 10**4, noise = 0.4, factor =0.001)

print("A")
X_pp_train, X_conf_train, y_pp_train, y_conf_train  = train_test_split(X,
                                                                      y,
                                                                      test_size=0.66,
                                                                      random_state=10)

print("B")
X_conf_train, X_cal, y_conf_train, y_cal = train_test_split(X_conf_train,
                                                           y_conf_train,
                                                           test_size=0.5,
                                                           random_state=0)

print("C")
X_cal, X_test, y_cal,  y_test, = train_test_split(X_cal,
                                                 y_cal,
                                                 test_size=0.5,
                                                 random_state=0)

print("D")
point_pred_model = RandomForestClassifier(random_state =0,
                                         n_estimators =800,
                                         n_jobs =-1)

print("E")
point_pred_model.fit(X_pp_train,
                    y_pp_train)

print("F")
macest_model = cl_mod.ModelWithConfidence(point_pred_model,
                                      X_conf_train,
                                      y_conf_train)
print("G")
macest_model.fit(X_cal, y_cal)

print("H")
conf_preds = macest_model.predict_confidence_of_point_prediction(X_test)
print(conf_preds)
