import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.stats
import warnings

from sklearn import svm
from itertools import compress
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import VarianceThreshold


train_data = pd.read_csv("../amex/train_data.csv")


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

train_data.isnull().sum()
train = train_data.groupby("customer_ID").tail(1)
train = train.set_index(["customer_ID"])

train.drop(["S_2"], axis=1, inplace=True)
train.shape

nan_cols = [
    "P_2",
    "B_2",
    "D_41",
    "D_84",
    "B_30",
    "B_33",
    "S_22",
    "S_23",
    "S_26",
    "B_36",
    "B_38",
    "D_124",
    "B_41",
    "D_140",
    "D_144",
]

num_cols = [
    "P_2",
    "D_39",
    "B_1",
    "B_2",
    "R_1",
    "S_3",
    "D_41",
    "B_3",
    "D_42",
    "D_43",
    "D_44",
    "B_4",
    "D_45",
    "B_5",
    "R_2",
    "D_46",
    "D_47",
    "D_48",
    "D_49",
    "B_6",
    "B_7",
    "B_8",
    "D_50",
    "D_51",
    "B_9",
    "R_3",
    "D_52",
    "P_3",
    "B_10",
    "D_53",
    "S_5",
    "B_11",
    "S_6",
    "D_54",
    "R_4",
    "S_7",
    "B_12",
    "S_8",
    "D_55",
    "D_56",
    "B_13",
    "R_5",
    "D_58",
    "S_9",
    "B_14",
    "D_59",
    "D_60",
    "D_61",
    "B_15",
    "S_11",
    "D_62",
    "D_65",
    "B_16",
    "B_17",
    "B_18",
    "B_19",
    "D_66",
    "B_20",
    "D_68",
    "S_12",
    "R_6",
    "S_13",
    "B_21",
    "D_69",
    "B_22",
    "D_70",
    "D_71",
    "D_72",
    "S_15",
    "B_23",
    "D_73",
    "P_4",
    "D_74",
    "D_75",
    "D_76",
    "B_24",
    "R_7",
    "D_77",
    "B_25",
    "B_26",
    "D_78",
    "D_79",
    "R_8",
    "R_9",
    "S_16",
    "D_80",
    "R_10",
    "R_11",
    "B_27",
    "D_81",
    "D_82",
    "S_17",
    "R_12",
    "B_28",
    "R_13",
    "D_83",
    "R_14",
    "R_15",
    "D_84",
    "R_16",
    "B_29",
    "B_30",
    "S_18",
    "D_86",
    "D_87",
    "R_17",
    "R_18",
    "D_88",
    "B_31",
    "S_19",
    "R_19",
    "B_32",
    "S_20",
    "R_20",
    "R_21",
    "B_33",
    "D_89",
    "R_22",
    "R_23",
    "D_91",
    "D_92",
    "D_93",
    "D_94",
    "R_24",
    "R_25",
    "D_96",
    "S_22",
    "S_23",
    "S_24",
    "S_25",
    "S_26",
    "D_102",
    "D_103",
    "D_104",
    "D_105",
    "D_106",
    "D_107",
    "B_36",
    "B_37",
    "R_26",
    "R_27",
    "B_38",
    "D_108",
    "D_109",
    "D_110",
    "D_111",
    "B_39",
    "D_112",
    "B_40",
    "S_27",
    "D_113",
    "D_114",
    "D_115",
    "D_116",
    "D_117",
    "D_118",
    "D_119",
    "D_120",
    "D_121",
    "D_122",
    "D_123",
    "D_124",
    "D_125",
    "D_126",
    "D_127",
    "D_128",
    "D_129",
    "B_41",
    "B_42",
    "D_130",
    "D_131",
    "D_132",
    "D_133",
    "R_28",
    "D_134",
    "D_135",
    "D_136",
    "D_137",
    "D_138",
    "D_139",
    "D_140",
    "D_141",
    "D_142",
    "D_143",
    "D_144",
    "D_145",
]

columns_to_load = [
    "P_2",
    "D_39",
    "R_1",
    "D_41",
    "B_3",
    "D_44",
    "B_4",
    "D_45",
    "B_5",
    "R_2",
    "D_47",
    "B_6",
    "B_8",
    "D_51",
    "B_9",
    "B_10",
    "S_5",
    "S_6",
    "S_8",
    "R_5",
    "D_60",
    "D_61",
    "D_62",
    "D_65",
    "B_19",
    "D_68",
    "S_12",
    "R_6",
    "S_13",
    "B_21",
    "D_69",
    "B_22",
    "D_70",
    "D_71",
    "D_72",
    "P_4",
    "B_24",
    "R_7",
    "B_26",
    "D_78",
    "D_79",
    "R_8",
    "S_16",
    "R_10",
    "D_81",
    "S_17",
    "B_28",
    "D_83",
    "R_14",
    "D_84",
    "R_16",
    "B_30",
    "R_20",
    "D_92",
    "S_23",
    "S_25",
    "S_26",
    "D_102",
    "D_107",
    "R_27",
    "B_38",
    "D_112",
    "B_40",
    "D_113",
    "D_114",
    "D_115",
    "D_117",
    "D_120",
    "D_121",
    "D_122",
    "D_124",
    "D_125",
    "D_126",
    "D_127",
    "D_128",
    "D_129",
    "B_41",
    "D_130",
    "D_131",
    "D_63_CL",
    "D_63_CO",
    "D_63_CR",
    "D_64_O",
    "D_64_R",
    "D_64_U",
]

# Pre-processing

# Impute
imp = SimpleImputer(missing_values=-1, strategy="most_frequent")
imp.fit_transform(train_data[nan_cols])


# Outlier Removal
for col in train_data[num_cols]:
    Q1 = np.percentile(train_data[col], 25, interpolation="midpoint")

    Q3 = np.percentile(train_data[col], 75, interpolation="midpoint")
    IQR = Q3 - Q1

    upper = np.where(train_data[col] >= (Q3 + 1.5 * IQR))
    lower = np.where(train_data[col] <= (Q1 - 1.5 * IQR))
    train_data.drop(upper[0], inplace=True)
    train_data.drop(lower[0], inplace=True)

# Onehot Encoding
train_D63 = pd.get_dummies(train[["D_63"]])
train = pd.concat([train, train_D63], axis=1)
train = train.drop(["D_63"], axis=1)

train_D64 = pd.get_dummies(train[["D_64"]])
train = pd.concat([train, train_D64], axis=1)
train = train.drop(["D_64"], axis=1)

# Remove cols
train = train.dropna(axis=1, thresh=int(0.85 * len(train)))

# Remove Hightly correlated features
train_without_target = train.drop(["target"], axis=1)
cor_matrix = train_without_target.corr().abs()
upper_tri = cor_matrix.where(
    (
        np.triu(np.ones(cor_matrix.shape), k=1)
        + np.tril(np.ones(cor_matrix.shape), k=-1)
    ).astype(bool)
)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
train_drop_highcorr = train_without_target.drop(to_drop, axis=1)
train_drop_highcorr.shape

# Standetixe
train = scaler = StandardScaler()
train = scaler.transform(num_cols)


def fs_variance(df, threshold: float = 0.05):
    features = list(df.columns)

    vt = VarianceThreshold(threshold=threshold)
    _ = vt.fit(df)

    feat_select = list(compress(features, vt.get_support()))

    return feat_select


columns_to_keep = fs_variance(train_drop_highcorr)
train_final = train[columns_to_keep]

train_final1 = train_final.join(train["target"])
x_train = train_final1.drop(["target"], axis=1)
y_train = train_final1["target"]


# Cross Validation
features = [f for f in train.columns if f != "customer_ID" and f != "target"]


def xgboost(random_state=1823, n_estimators=1500):
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        reg_lambda=50,
        min_child_samples=2400,
        num_leaves=95,
        colsample_bytree=0.19,
        max_bins=511,
        random_state=random_state,
    )


def svm():
    svm_clf = svm.SVC(decision_function_shape="ovo")
    return svm_clf


def knn():
    clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    return clf


score_list = []
y_pred_list = []
kf = StratifiedKFold(n_splits=2)
for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train["target"])):
    X_tr, X_va, y_tr, y_va, model = None, None, None, None, None
    start_time = datetime.datetime.now()
    X_tr = train.iloc[idx_tr][features]
    X_va = train.iloc[idx_va][features]
    y_tr = train["target"][idx_tr]
    y_va = train["target"][idx_va]

    model = xgboost()
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
    )
    X_tr, y_tr = None, None
    y_va_pred = model.predict_proba(X_va, raw_score=True)
    score = accuracy_score(y_va, y_va_pred)
    n_trees = model.best_iteration_
    if n_trees is None:
        n_trees = model.n_estimators
    score_list.append(score)

# train

x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
    x_train, y_train, test_size=0.25, random_state=26
)
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.15, subsample=0.5)
model.fit(x_train_split, y_train_split)
y_predict = model.predict(x_test_split)

test_data = pd.read_parquet("../amex/test_data.csv", columns=columns_to_load)
test = test_data.groupby("customer_ID").tail(1)
test = test.set_index(["customer_ID"])

test.drop(["S_2"], axis=1, inplace=True)
test_D63 = pd.get_dummies(test[["D_63"]])
test = pd.concat([test, test_D63], axis=1)
test = test.drop(["D_63"], axis=1)

test_D64 = pd.get_dummies(test[["D_64"]])
test = pd.concat([test, test_D64], axis=1)
test = test.drop(["D_64"], axis=1)
test_final = test[columns_to_keep]

y_test_predict = model.predict_proba(test_final)
y_predict_final = y_test_predict[:, 1]

submission = pd.DataFrame(
    {"customer_ID": test_final.index, "prediction": y_predict_final}
)

submission.to_csv("submission13.csv", index=False)
