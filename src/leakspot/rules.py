"""Rule catalogue and the library knowledge the scanner relies on.

Taxonomy codes refer to Kapoor & Narayanan (2023), "Leakage and the
reproducibility crisis in machine-learning-based science", Patterns 4(9).
"""

from __future__ import annotations

from dataclasses import dataclass

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    summary: str
    fix: str
    taxonomy: str


RULES: dict[str, Rule] = {r.id: r for r in [
    Rule("LS101", "preprocessing-before-split",
         "A preprocessing step learns statistics from data that is later split "
         "into train and test, so test-set information reaches training.",
         "Split first and fit preprocessing on the training set only, or put the "
         "step inside a scikit-learn Pipeline.",
         "L1.2 pre-processing on train and test / L1.3 feature selection on train and test"),
    Rule("LS102", "resampling-before-split",
         "Over/under-sampling (e.g. SMOTE) runs before the split, so synthetic or "
         "duplicated rows derived from test samples end up in training.",
         "Resample only the training set after splitting, ideally with an "
         "imbalanced-learn Pipeline so it also stays inside each CV fold.",
         "L1.2 pre-processing on train and test"),
    Rule("LS103", "fit-on-test-data",
         "A model or preprocessing step is fitted on data that contains the test "
         "set, so the test set no longer measures generalisation.",
         "Only call fit / fit_transform / fit_resample on training data; use "
         "transform / predict on test data.",
         "L1.1 no clean test set / L1.2 pre-processing on train and test"),
    Rule("LS104", "preprocessing-outside-cv",
         "Cross-validation or hyper-parameter search runs on data that was "
         "already transformed by a fitted step, so every validation fold has "
         "influenced the preprocessing.",
         "Wrap preprocessing and model in a Pipeline and pass the raw data to "
         "cross_val_score / GridSearchCV.",
         "L1.2 pre-processing on train and test"),
    Rule("LS105", "target-encoding-before-split",
         "A feature is built by aggregating the target (e.g. groupby(...)[target]"
         ".mean()) over rows that include the test set.",
         "Compute target statistics on the training set only (or out-of-fold), "
         "e.g. with sklearn.preprocessing.TargetEncoder inside a Pipeline.",
         "L1.2 pre-processing on train and test / L2 illegitimate features"),
    Rule("LS106", "shuffled-split-on-time-data",
         "The data looks time-stamped but train_test_split shuffles rows, so the "
         "model can train on the future and be tested on the past.",
         "If rows are time-ordered, use shuffle=False, a cut-off date or "
         "TimeSeriesSplit. Ignore this if the rows are independent of time.",
         "L3.1 temporal leakage"),
]}


def _cls(names: str, kind: str, severity: str) -> dict[str, tuple[str, str]]:
    return {n: (kind, severity) for n in names.split()}


# class name -> (kind, severity when fitted on train+test)
TRANSFORMERS: dict[str, tuple[str, str]] = {
    **_cls("StandardScaler MinMaxScaler RobustScaler MaxAbsScaler QuantileTransformer "
           "PowerTransformer KBinsDiscretizer SplineTransformer", "scale", "medium"),
    **_cls("SimpleImputer KNNImputer IterativeImputer MissingIndicator", "impute", "medium"),
    **_cls("PCA TruncatedSVD KernelPCA IncrementalPCA FastICA NMF FactorAnalysis "
           "UMAP TSNE Isomap", "reduce", "medium"),
    **_cls("TfidfVectorizer CountVectorizer TfidfTransformer", "text", "medium"),
    **_cls("OneHotEncoder OrdinalEncoder DictVectorizer", "encode", "low"),
    **_cls("SelectKBest SelectPercentile SelectFromModel RFE RFECV VarianceThreshold "
           "SequentialFeatureSelector SelectFpr SelectFdr SelectFwe "
           "GenericUnivariateSelect Boruta BorutaPy", "select", "high"),
    **_cls("SMOTE ADASYN BorderlineSMOTE SVMSMOTE KMeansSMOTE SMOTENC SMOTEN "
           "RandomOverSampler RandomUnderSampler SMOTEENN SMOTETomek NearMiss "
           "TomekLinks EditedNearestNeighbours ClusterCentroids", "resample", "high"),
    **_cls("TargetEncoder CatBoostEncoder WOEEncoder JamesSteinEncoder "
           "LeaveOneOutEncoder MEstimateEncoder GLMMEncoder", "target_encode", "high"),
}

# Fitting these on all data is harmless or standard practice.
IGNORED_TRANSFORMERS = set("LabelEncoder LabelBinarizer MultiLabelBinarizer "
                           "FunctionTransformer PolynomialFeatures Normalizer "
                           "Binarizer HashingVectorizer".split())

PIPELINES = set("Pipeline make_pipeline ColumnTransformer make_column_transformer "
                "FeatureUnion make_union".split())
SEARCHES = set("GridSearchCV RandomizedSearchCV HalvingGridSearchCV "
               "HalvingRandomSearchCV BayesSearchCV TuneSearchCV".split())
CV_FUNCS = set("cross_val_score cross_validate cross_val_predict learning_curve "
               "validation_curve permutation_test_score".split())
FIT_METHODS = {"fit", "fit_transform", "fit_resample", "partial_fit", "fit_predict"}
STAT_METHODS = {"mean", "std", "var", "median", "min", "max", "quantile", "mode"}
STAT_FUNCS = {"nanmean", "nanmedian", "nanstd", "nanvar", "average", "percentile",
              "nanpercentile"}
HIGH_RISK_KINDS = {"select", "resample", "target_encode"}
