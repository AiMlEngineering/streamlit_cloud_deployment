# import library and module
import sys
from pathlib import Path

from category_encoders.binary import BinaryEncoder
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer,
                                       CategoricalImputer, EndTailImputer,
                                       MeanMedianImputer, RandomSampleImputer)
from feature_engine.outliers import ArbitraryOutlierCapper, Winsorizer
from feature_engine.selection import (DropConstantFeatures,
                                      DropDuplicateFeatures, DropFeatures)
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, Normalizer, RobustScaler,
                                   StandardScaler)
from sklearn.tree import DecisionTreeRegressor

from sources.feature_engineering.sources import \
    feature_engineering as feat_engin_src
from sources.models_and_pipelines.sources import \
    models_and_pipelines as mod_and_pipe_src
from sources.pipelines.sources import house_preprocessors as h_pp
from sources.pipelines.sources import \
    house_preprocessors_data_prep_clean as h_pp_data_prep_clean

# for reproducibility, split size
seed = 0
split_size = 0.3

# fraction of outliers in data set
outlier_fraction = 0.1

# where the command is executed
sys.path.append(str(Path().absolute()))

# # to print the path of execute command
# pprint(sys.path)


# Set imputer associated to each method
imputer_bayes = IterativeImputer(
    estimator=BayesianRidge(), max_iter=10, random_state=seed
)
imputer_knn = IterativeImputer(
    estimator=KNeighborsRegressor(n_neighbors=5), max_iter=10, random_state=seed
)
imputer_nonLin = IterativeImputer(
    estimator=DecisionTreeRegressor(max_features="sqrt", random_state=seed),
    max_iter=500,
    random_state=seed,
)
imputer_missForest = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=seed),
    max_iter=100,
    random_state=seed,
)
imputer_rf = IterativeImputer(
    estimator=RandomForestRegressor(max_depth=2, random_state=seed),
    max_iter=100,
    random_state=seed,
)

# load all neccessary variable
# about dictionnary
numtocat_features_dict = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_with_unique_dictionnary.pkl"
)
cattonum_features_dict = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="unique_with_numerical_dictionnary.pkl"
)

# about features to drop
irrelevant_features = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="irrelevant_features.pkl"
)
duplicate_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="duplicated_features.pkl"
)
constante_cat_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="constant_categorical_features.pkl"
)
constante_num_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="constant_numerical_features.pkl"
)
num_var_with_high_multicolinearity = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="numerical_variable_with_high_multicolinearity.pkl",
)
num_var_with_law_multicolinearity = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="numerical_variable_with_less_multicolinearity.pkl",
)


# about numerical variable and missing value
num_var_original_data = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_original_data.pkl"
)
num_var_prep_clean_data = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_prep_clean_data.pkl"
)
num_var_for_somme = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_for_somme.pkl"
)
num_var_for_product = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_for_product.pkl"
)
num_var_for_ratio = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_for_ratio.pkl"
)
num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable.pkl"
)
num_var_to_drop_for_linear_model = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="numerical_variable_to_drop_for_linear_model.pkl",
)
num_var_to_transform_for_linear_model = (
    feat_engin_src.get_variable_list_by_loading_file(
        file_path="./variables/",
        file_name="numerical_variable_to_transform_for_linear_model.pkl",
    )
)

# about categorical variable and missing value
cat_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="categorical_variable.pkl"
)
miss_val = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="value_with_nan.pkl"
)
miss_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="variable_with_nan.pkl"
)

# about numerical or categorical variable to treat
casted_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_casted.pkl"
)
arb_num_val = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="arbitrary_number_value.pkl"
)
noskew_cat_var_for_grouping = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="categorical_variable_for_grouping_noskewed_var.pkl",
)
skew_cat_var_for_grouping = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="categorical_variable_for_grouping_skewed_var.pkl",
)

# about continuous and discrete variable
num_var_cont = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="continuous_numerical_variable.pkl"
)
num_var_disc = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="discrete_numerical_variable.pkl"
)

# about normal and skewed distribution variable
norm_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="normal_dist_numerical_variable.pkl"
)
skew_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="skewed_dist_numerical_variable.pkl"
)
num_var_with_low_variation_for_iqr = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="numerical_variable_with_low_variation_for_IQR_method.pkl",
)
num_var_with_high_variation_for_iqr = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="numerical_variable_with_high_variation_for_IQR_method.pkl",
)

# about variable contains outliers
outliers_with_IQR_for_skew_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="outliers_detected_with_IQR_method_for"
    "_skewed_distribution_numerical_variable.pkl",
)
outliers_with_Gaussian_for_normal_num_var = (
    feat_engin_src.get_variable_list_by_loading_file(
        file_path="./variables/",
        file_name="outliers_detected_with_Gaussian_method_for"
        "_normal_distribution_numerical_variable.pkl",
    )
)
outliers_with_Quantiles_for_normal_num_var = (
    feat_engin_src.get_variable_list_by_loading_file(
        file_path="./variables/",
        file_name="outliers_detected_with_Quantiles_method_for"
        "_normal_distribution_numerical_variable.pkl",
    )
)

# about numerical variable
miss_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="numerical_variable_with_nan.pkl"
)
miss_skew_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="skewed_dist_numerical_variable_with_nan.pkl"
)
miss_norm_num_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="normal_dist_numerical_variable_with_nan.pkl"
)

# about categorical variable
cat_var_original_data = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="categorical_variable_original_data.pkl"
)
cat_var_prep_clean_data = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="categorical_variable_prep_clean_data.pkl"
)
miss_cat_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="categorical_variable_with_nan.pkl"
)
cat_var_with_rare_labels = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="categorical_variable_with_Rare_Label_To_ReGroup.pkl",
)

# about target variable
targ_var_original_data = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="target_variable_original_data.pkl"
)
targ_var_prep_clean_data = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="target_variable_prep_clean_data.pkl"
)

# about embedding variable
var_to_embedding = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="variable_to_embedding.pkl"
)
USE_embed_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="USE_embed_variable.pkl"
)
IS_embed_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="USE_embed_variable.pkl"
)
DV_embed_var = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/", file_name="DV_embed_variable.pkl"
)
# LASER_embed_var = feat_engin_src.get_variable_list_by_loading_file(
#     file_path="./variables/", file_name="LASER_embed_variable.pkl"
# )

# final feature selected and remove for ML model in production
final_feature_selected_byLogReg = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="logreg_feature_selected_by_confirm_selected_feature.pkl",
)
final_feature_removed_byLogReg = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="logreg_feature_removed_by_confirm_selected_feature.pkl",
)

# about the machine learning model
# load the model want to use
bagg_logreg_model_default = mod_and_pipe_src.load_model_from_directory(
    directory="./models/default", file_name="bagg_logreg_default.pkl"
)
bagg_logreg_model_tuned = mod_and_pipe_src.load_model_from_directory(
    directory="./models/tuned", file_name="bagg_logreg_tuned.pkl"
)

# model, thresholds and feature selected for stacking estimator building stage
logreg = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="mlflow_bagg_def_logreg_tuned_tpe_class_1.pkl",
)
logreg_thresholds = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./models/choose",
    file_name="threshold_mlflow_bagg_def_logreg_tuned_tpe_class_1.pkl",
)
logreg_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="logreg_feature_selected_by_confirm_selected_feature.pkl",
)

guaproclas = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose", file_name="mlflow_guaproclas_def_class_1.pkl"
)
guaproclas_thresholds = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./models/choose", file_name="threshold_mlflow_guaproclas_def_class_1.pkl"
)
guaproclas_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="guaproclas_feature_selected_by_confirm_selected_feature.pkl",
)

lindiscana = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="mlflow_bagg_def_lindiscana_tuned_tpe_class_1.pkl",
)
lindiscana_thresholds = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./models/choose",
    file_name="threshold_mlflow_bagg_def_lindiscana_tuned_tpe_class_1.pkl",
)
lindiscana_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="lindiscana_feature_selected_by_confirm_selected_feature.pkl",
)

perceptron = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="bagg_tuned_tpe_perceptron_tuned_rand_class_1.pkl",
)
perceptron_thresholds = None
perceptron_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="perceptron_feature_selected_by_confirm_selected_feature.pkl",
)

quadiscana = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose", file_name="mlflow_quadiscana_def_class_1.pkl"
)
quadiscana_thresholds = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./models/choose", file_name="threshold_mlflow_quadiscana_def_class_1.pkl"
)
quadiscana_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="quadiscana_feature_selected_by_confirm_selected_feature.pkl",
)

sgdclas_noproba = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose", file_name="mlflow_sgdclas_noproba_def_class_1.pkl"
)
sgdclas_noproba_thresholds = None
sgdclas_noproba_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="sgdclas_noproba_feature_selected_by_confirm_selected_feature.pkl",
)

sgdclas_proba = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose", file_name="sgdclas_proba_tuned_tpe_class_1.pkl"
)
sgdclas_proba_thresholds = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./models/choose",
    file_name="threshold_sgdclas_proba_tuned_tpe_class_1.pkl",
)
sgdclas_proba_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="sgdclas_proba_feature_selected_by_confirm_selected_feature.pkl",
)

# mlp = mod_and_pipe_src.load_ann_model_from_directory(
#     file_path='./models/choose',
#     model_name='mlflow_mlp_tuned_rand_class_1.keras'
# )
# mlp_thresholds = mod_and_pipe_src.load_model_from_directory(
#     directory='./models/choose',
#     file_name='threshold_mlflow_mlp_tuned_rand_class_1.pkl'
# )
# mlp_feature = feat_engin_src.get_variable_list_by_loading_file(
#     file_path="./variables/",
#     file_name="mlp_feature_selected_by_confirm_selected_feature.pkl"
# )

mlpclas = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="bagg_default_mlpclassifier_tuned_tpe_class_1.pkl",
)
mlpclas_thresholds = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="threshold_bagg_default_mlpclassifier_tuned_tpe_class_1.pkl",
)
mlpclas_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="mlpclassifier_feature_selected_by_confirm_selected_feature.pkl",
)

mlp_classifier = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="mlflow_bagg_tuned_tpe_mlpclassifier_stacking_tuned_tpe_class_1.pkl",
)
mlp_classifier_thresholds = mod_and_pipe_src.load_model_from_directory(
    directory="./models/choose",
    file_name="threshold_mlflow_bagg_tuned_tpe_mlpclassifier_stacking_tuned_tpe_class_1.pkl",
)
mlp_classifier_feature = feat_engin_src.get_variable_list_by_loading_file(
    file_path="./variables/",
    file_name="mlpclassifier_stacking_feature_selected_by_confirm_selected_feature.pkl",
)

# variables selected, estimators, threshold and final estimator for Stacking estimator
estimators_variables_selected = [
    logreg_feature,
    guaproclas_feature,
    lindiscana_feature,
    perceptron_feature,
    quadiscana_feature,
    sgdclas_noproba_feature,
    sgdclas_proba_feature,
    mlpclas_feature,
]
estimators = [
    logreg,
    guaproclas,
    lindiscana,
    perceptron,
    quadiscana,
    sgdclas_noproba,
    sgdclas_proba,
    mlpclas,
]
estimators_names = [
    "logreg",
    "guaproclas",
    "lindiscana",
    "perceptron",
    "quadiscana",
    "sgdclas_noproba",
    "sgdclas_proba",
    "mlpclas",
]
estimators_thresholds = [
    logreg_thresholds,
    guaproclas_thresholds,
    lindiscana_thresholds,
    perceptron_thresholds,
    quadiscana_thresholds,
    sgdclas_noproba_thresholds,
    sgdclas_proba_thresholds,
    mlpclas_thresholds,
]

final_estimators_variables_selected = [
    mlp_classifier_feature,
]
final_estimators = [
    mlp_classifier,
]
final_estimators_names = [
    "mlpclassifier",
]
final_estimators_thresholds = [
    mlp_classifier_thresholds,
]


# set up the pipeline
# using Feature-engine open source Library for building transformers

# ===== BUILDING PIPELINES FOR DATA ANALYSIS =====
pipe_add_nan_rows_to_train_set = Pipeline(
    [
        # for adding artificial NaN rows to train seet in order to be able to
        # to predict in production any feature set comes with NaN values
        (
            "add_artificial_NaN_rows_to_train_set",
            h_pp.AddArtificialNaNRowsToTrainSetTransform(),
        ),
    ]
)

pipe_rename_homedest_column = Pipeline(
    [
        # for checking and rename the name of column "home.dest" to "homedest"
        (
            "check_rename_certain_column_name",
            h_pp_data_prep_clean.CheckAndRenameHomedestColumnNameTransform(),
        ),
    ]
)

pipe_replace_missing_value_byNaN = Pipeline(
    [
        # to searching missing value in data set and replace them by Nan
        (
            "replace_missing_value_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(missing_values_list=miss_val),
        ),
    ]
)

pipe_cast_numerical_variable = Pipeline(
    [
        # to cast some numerical variable as float
        (
            "casted_some_numerical_variable",
            h_pp.CastingCertainNumericalVariableAsFloatTransform(
                casted_variable_list=casted_num_var
            ),
        ),
    ]
)

pipe_assign_variable_type_to_original_feature = Pipeline(
    [
        # to check and assign the right type to all variable in data set
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=num_var_original_data,
                categorical_variable_list=cat_var_original_data,
            ),
        ),
    ]
)

pipe_assign_variable_type = Pipeline(
    [
        # to check and assign the right type to all variable in data set
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=num_var_prep_clean_data,
                categorical_variable_list=cat_var_prep_clean_data,
            ),
        ),
    ]
)

pipe_extract_all_title_from_variable_name = Pipeline(
    [
        # to extract all title contains in the variable "name"
        (
            "extract_all_title_in_variable_name",
            h_pp_data_prep_clean.ExtractionAllTitleFromTheNameTransform(),
        ),
    ]
)

pipe_extract_substring_from_variable_ticket = Pipeline(
    [
        # to extract all substring contains in the variable "ticket"
        (
            "extract_substring_from_variable_ticket",
            h_pp_data_prep_clean.ExtractionSubstringFromTicketTransform(),
        ),
    ]
)

pipe_extract_substring_from_variable_cabin = Pipeline(
    [
        # to extract all substring contains in the variable "cabin"
        (
            "extract_substring_from_variable_cabin",
            h_pp_data_prep_clean.ExtractionSubstringFromCabinTransform(),
        ),
    ]
)

pipe_extract_substring_from_variable_homedest = Pipeline(
    [
        # to extract all substring contains in the variable "homedest"
        (
            "extract_substring_from_variable_homedest",
            h_pp_data_prep_clean.ExtractionSubstringFromHomeDestTransform(),
        ),
    ]
)


# ===== BUILDING PIPELINES TO IMPUT MISSING VALUES FOR NUMERICAL VARIABLE IN DATASET =====
pipe_imput_numvar_MCAR_with_mean_median = Pipeline(
    [
        # for variable with normal distribution
        (
            "nd_mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=miss_norm_num_var,
            ),
        ),
        # # for variable with skewed distribution
        # (
        #     "sd_median_imputation",
        #     MeanMedianImputer(
        #         imputation_method="median",
        #         variables=miss_skew_num_var,
        #     ),
        # ),
    ]
)

pipe_imput_numvar_MAR_with_mean_median_plus_missing_indicator = Pipeline(
    [
        # for variable with normal distribution
        (
            "nd_missing_indicator",
            AddMissingIndicator(variables=miss_norm_num_var),
        ),
        (
            "nd_mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=miss_norm_num_var,
            ),
        ),
        # # for variable with skewed distribution
        # (
        #     "sd_missing_indicator",
        #     AddMissingIndicator(
        #         variables=miss_skew_num_var
        #     ),
        # ),
        # (
        #     "sd_median_imputation",
        #     MeanMedianImputer(
        #         imputation_method="median",
        #         variables=miss_skew_num_var,
        #     ),
        # ),
    ]
)

pipe_imput_numvar_MAR_with_random_sample = Pipeline(
    [
        # for all variable
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_num_var, random_state=seed),
        ),
    ]
)

pipe_imput_numvar_MAR_with_mean_median_plus_missing_indicator_plus_grouping_variable = (
    Pipeline(
        [
            # for variable with normal distribution
            (
                "nd_missing_indicator",
                AddMissingIndicator(variables=miss_norm_num_var),
            ),
            (
                "nd_grouping_variable",
                h_pp.GroupingVariableTransform(
                    variables=miss_norm_num_var,
                    categorical_variable_for_grouping=noskew_cat_var_for_grouping,
                ),
            ),
            (
                "nd_mean_imputation",
                MeanMedianImputer(
                    imputation_method="mean",
                    variables=miss_norm_num_var,
                ),
            ),
            # # for variable with skewed distribution
            # (
            #     "sd_missing_indicator",
            #     AddMissingIndicator(
            #         variables=miss_skew_num_var
            #     ),
            # ),
            # (
            #     "sd_grouping_variable",
            #     h_pp.GroupingVariableTransform(
            #         variables=miss_skew_num_var,
            #         categorical_variable_for_grouping=skew_cat_var_for_grouping
            #     ),
            # ),
            # (
            #     "sd_median_imputation",
            #     MeanMedianImputer(
            #         imputation_method="median",
            #         variables=miss_skew_num_var,
            #     ),
            # ),
        ]
    )
)

pipe_imput_numvar_MAR_with_MICE_using_BayesianRidge = Pipeline(
    [
        # for numerical variable
        (
            "MICE_and_MissForest_imputation",
            h_pp.MICEImputationTransform(variables=miss_num_var, imputer=imputer_bayes),
        ),
    ]
)

pipe_imput_numvar_MAR_with_MICE_using_KNeighborsRegressor = Pipeline(
    [
        # for numerical variable
        (
            "MICE_and_MissForest_imputation",
            h_pp.MICEImputationTransform(variables=miss_num_var, imputer=imputer_knn),
        ),
    ]
)

pipe_imput_numvar_MAR_with_MICE_using_DecisionTreeRegressor = Pipeline(
    [
        # for numerical variable
        (
            "MICE_and_MissForest_imputation",
            h_pp.MICEImputationTransform(
                variables=miss_num_var, imputer=imputer_nonLin
            ),
        ),
    ]
)

pipe_imput_numvar_MAR_with_MICE_using_ExtraTreesRegressor = Pipeline(
    [
        # for numerical variable
        (
            "MICE_and_MissForest_imputation",
            h_pp.MICEImputationTransform(
                variables=miss_num_var, imputer=imputer_missForest
            ),
        ),
    ]
)

pipe_imput_numvar_MAR_with_MICE_using_RandomForestRegressor = Pipeline(
    [
        # for numerical variable
        (
            "MICE_and_MissForest_imputation",
            h_pp.MICEImputationTransform(variables=miss_num_var, imputer=imputer_rf),
        ),
    ]
)

pipe_imput_numvar_MNAR_with_arbitrary_value = Pipeline(
    [
        # for all variable
        (
            "arbitrary_value",
            ArbitraryNumberImputer(
                arbitrary_number=arb_num_val, variables=miss_num_var
            ),
        ),
    ]
)

pipe_imput_numvar_MNAR_with_end_of_tail = Pipeline(
    [
        # for variable with normal distribution
        (
            "gaussian_approximation_imputation",
            EndTailImputer(
                imputation_method="gaussian", tail="right", variables=miss_norm_num_var
            ),
        ),
        # # for variable with skewed distribution
        # (
        #     "inter-quartile_range_imputation",
        #     EndTailImputer(
        #         imputation_method="iqr",
        #         tail="right",
        #         variables=miss_skew_num_var
        #     ),
        # ),
    ]
)


# ===== BUILDING PIPELINES TO IMPUT MISSING VALUES FOR CATEGORICAL VARIABLE IN DATASET =====
pipe_imput_catvar_MCAR_with_frequent_category = Pipeline(
    [
        # for variable MCAR
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
    ]
)

pipe_imput_catvar_MAR_with_frequent_category_plus_missing_indicator = Pipeline(
    [
        # for variable MAR
        (
            "missing_indicator",
            AddMissingIndicator(variables=miss_cat_var),
        ),
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
    ]
)

pipe_imput_catvar_MAR_with_random_sample_imputation = Pipeline(
    [
        # for variable MAR
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_cat_var, random_state=seed),
        ),
    ]
)

pipe_imput_catvar_MNAR_with_arbitrary_or_missing_category = Pipeline(
    [
        # for variable MAR
        (
            "arbitrary_or_missing_category_imputation",
            CategoricalImputer(variables=miss_cat_var),
        ),
    ]
)


# ===== BUILDING PIPELINES TO GENERATE PRODUCT AND RATIO VARIABLE IN DATASET =====
pipe_generate_somme_product_and_ratio_variable = Pipeline(
    [
        # to check and assign the right type to all variable in data set
        (
            "somme_variable_generation",
            h_pp.GenerateSommeVariableTransform(variable_list=num_var_for_somme),
        ),
        (
            "product_variable_generation",
            h_pp.GenerateProductVariableTransform(variable_list=num_var_for_product),
        ),
        (
            "ratio_variable_generation",
            h_pp.GenerateRatioVariableTransform(variable_list=num_var_for_ratio),
        ),
    ]
)


# ===== BUILDING PIPELINES FOR OUTLIERS DETECTING + INDICATOR ADDING + CAPPING OR REMOVING =====
pipe_add_indicator_and_capping_outliers = Pipeline(
    [
        # for adding artificial NaN rows to train seet in order to be able to
        # to predict in production any feature set comes with NaN values
        (
            "add_artificial_NaN_rows_to_train_set",
            h_pp.AddArtificialNaNRowsToTrainSetTransform(),
        ),
    ]
)


# ===== BUILDING PIPELINES TO IMPROVE DATA FOR LINEAR REGRESSION MODEL =====
pipe_transform_data_for_improvement_of_linear_regression_model = Pipeline(
    [
        # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_Gaussian_outliers_indicator_for_capping",
        #     h_pp.AddGaussianOutliersIndicatorTransform(
        #         variable_list=outliers_with_Gaussian_for_normal_num_var,
        #         fold=3.0
        #     ),
        # ),
        # # Outliers handling
        # # to remove outliers in the variable with normal distribution
        # (
        #     "outliers_removing_for_normal_variable_with_gaussian_method",
        #     OutlierTrimmer(
        #         variables=outliers_with_Gaussian_for_normal_num_var,
        #         capping_method="gaussian",
        #         tail="both",
        #         fold=3.0,
        #     ),
        # ),
        # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_Quantiles_outliers_indicator_for_capping",
        #     h_pp.AddQuantilesOutliersIndicatorTransform(
        #         variable_list=outliers_with_Quantiles_for_normal_num_var
        #     ),
        # ),
        # # Outliers handling
        # # to remove outliers in the variable with quantiles
        # (
        #     "outliers_removing_for_normal_variable_with_quantiles_method",
        #     OutlierTrimmer(
        #         variables=outliers_with_Quantiles_for_normal_num_var,
        #         capping_method="quantiles",
        #         tail="both",
        #         fold=0.05,
        #     ),
        # ),
        # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_Iqr_outliers_indicator_for_capping",
        #     h_pp.AddIqrOutliersIndicatorTransform(
        #         variable_list=outliers_with_IQR_for_skew_num_var,
        #         fold=1.5
        #     ),
        # ),
        # # Outliers handling
        # # to remove outliers in the variable with skewed distribution
        # (
        #     "outliers_removing_for_skewed_variable_with_IQR_method",
        #     OutlierTrimmer(
        #         variables=outliers_with_IQR_for_skew_num_var,
        #         capping_method="iqr",
        #         tail="both",
        #         fold=1.5,
        #     ),
        # ),
        # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_Gaussian_outliers_indicator_for_capping",
        #     h_pp.AddGaussianOutliersIndicatorTransform(
        #         variable_list=outliers_with_Gaussian_for_normal_num_var,
        #         fold=3.0
        #     ),
        # ),
        # # Outliers handling
        # # to cap outliers in the variable with normal distribution
        # (
        #     "outliers_capping_for_normal_variable_with_gaussian_method",
        #     Winsorizer(
        #         variables=outliers_with_Gaussian_for_normal_num_var,
        #         capping_method="gaussian",
        #         tail="both",
        #         fold=3.0,
        #     ),
        # ),
        # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_Quantiles_outliers_indicator_for_capping",
        #     h_pp.AddQuantilesOutliersIndicatorTransform(
        #         variable_list=outliers_with_Quantiles_for_normal_num_var
        #     ),
        # ),
        # # Outliers handling
        # # to cap outliers in the variable with normal distribution
        # (
        #     "outliers_capping_for_normal_variable_with_quantiles_method",
        #     Winsorizer(
        #         variables=outliers_with_Quantiles_for_normal_num_var,
        #         capping_method="quantiles",
        #         tail="both",
        #         fold=0.05,
        #     ),
        # ),
        # # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_Iqr_outliers_indicator_for_capping",
        #     h_pp.AddIqrOutliersIndicatorTransform(
        #         variable_list=outliers_with_IQR_for_skew_num_var,
        #         fold=3.0
        #     ),
        # ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=outliers_with_IQR_for_skew_num_var,
                capping_method="iqr",
                tail="both",
                fold=3.0,
            ),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_minimum_outliers_indicator_for_arbitrary_capping",
            h_pp.AddMinimumArbitraryOutliersIndicatorTransform(
                max_capping_dict=None, min_capping_dict={"age": 20, "fare": 30}
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with arbitrary-capping method
        (
            "minimum_outliers_capping_for_all_variable_with_arbitrary_capping_method",
            ArbitraryOutlierCapper(
                max_capping_dict=None,
                min_capping_dict={"age": 20, "fare": 30},
            ),
        ),
        # # Outliers handling
        # # Add Outliers indicator
        # (
        #     "add_maximum_outliers_indicator_for_arbitrary_capping",
        #     h_pp.AddMaximumArbitraryOutliersIndicatorTransform(
        #         max_capping_dict={
        #             'pclass': 2, 'body': 5
        #         },
        #         min_capping_dict=None
        #     ),
        # ),
        # # Outliers handling
        # # to cap outliers in the variable with arbitrary-capping method
        # (
        #     "maximum_outliers_capping_for_all_variable_with_arbitrary_capping_method",
        #     ArbitraryOutlierCapper(
        #         max_capping_dict={'age': 50, 'fare': 200},
        #         min_capping_dict=None,
        #     ),
        # ),
        # # Outliers handling
        # Add Outliers indicator
        # (
        #     "add_both_end_outliers_indicator_for_arbitrary_capping",
        #     h_pp.AddArbitraryOutliersIndicatorTransform(
        #         max_capping_dict={
        #             'body': 7, 'parch': 0.7
        #         },
        #         min_capping_dict={
        #             'body': 5, 'parch': 0.2
        #         }
        #     ),
        # ),
        # # Outliers handling
        # # to cap outliers in the variable with arbitrary-capping method
        # (
        #     "both_ends_outliers_capping_for_all_variable_with_arbitrary_capping_method",
        #     ArbitraryOutlierCapper(
        #         max_capping_dict={
        #             'age': 50, 'fare': 200
        #         },
        #         min_capping_dict={
        #             'age': 10, 'fare': 100
        #         }
        #     ),
        # ),
        # # Outliers handling
        # # to apply discretization method to continuous variable
        # (
        #     "variable_discretization_with_arbitrary_discretizer_method",
        #     ArbitraryDiscretiser(
        #         binning_dict = {
        #             "age": [0, 18, 25, 40, 80],
        #             "fare": [0, 20, 50, 100, 600]},
        #     ),
        # ),
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=num_var),
        ),
        # # Convert variable with skewed distribution into normal distribution
        # # to apply yeo-johnson transformation to continuous and discrete variable
        # (
        #     "yeojohnson_transformation_application",
        #     YeoJohnsonTransformer(
        #         variables=num_var
        #     ),
        # ),
        # # Convert variable with skewed distribution into normal distribution
        # # to apply discretization+monotonic encoding method to continuous variable
        # (
        #     "variable_discretization_with_equal_frequency_discretizer_method",
        #     EqualFrequencyDiscretiser(
        #         q=10,
        #         variables=num_var_cont,
        #         return_object=True
        #     ),
        # ),
        # # Convert variable with skewed distribution into normal distribution
        # # to reorder the variable discretized by target
        # (
        #     "reorder_variable_discretized_by_target",
        #     OrdinalEncoder(
        #         encoding_method = 'ordered',
        #         variables=num_var_cont,
        #         ignore_format=True
        #     ),
        # ),
    ]
)

# set pipeline for apply mathematical transformation to data set
pipe_transform_num_feature = Pipeline(
    [
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=skew_num_var),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_Iqr_outliers_indicator_for_capping",
            h_pp.AddIqrOutliersIndicatorTransform(variable_list=num_var, fold=1.5),
        ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=num_var_with_high_variation_for_iqr,
                capping_method="iqr",
                tail="both",
                fold=1.5,
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with normal distribution
        (
            "outliers_capping_for_normal_variable_with_gaussian_method",
            Winsorizer(
                variables=num_var_with_low_variation_for_iqr,
                capping_method="gaussian",
                tail="both",
                fold=1.5,
            ),
        ),
    ]
)


# ===== BUILDING PIPELINES TO ENCODE CATEGORICAL VARIABLES =====
# Build Pipeline with sklearn and ColumnTransformer
# BinaryEncoder pipeline
pipe_encoder = Pipeline(
    steps=[
        ("binary_encoder", BinaryEncoder()),
    ]
)
# column transformer process
be_preprocessor = ColumnTransformer(
    transformers=[
        ("BE", pipe_encoder, cat_var),
    ],
    remainder="passthrough",
).set_output(transform="pandas")
# final pipeline
pipe_be_with_sklearn = Pipeline(
    steps=[
        ("binary_encoder_preprocessor", be_preprocessor),
    ]
)

pipe_encode_categorical_variable = Pipeline(
    [
        # to perform ancoding of categorical variable
        # Re-group rare labels into "Rare"
        (
            "regroup_rare_labels_into_Rare",
            RareLabelEncoder(
                tol=0.05,  # minimal percentage to be considered non-rare
                n_categories=2,  # minimal number of categories required to re-group rare categorie
                variables=cat_var_with_rare_labels,  # variables to re-group
            ),
        ),
        # # perform one-hot-encoding
        # (
        #     "one_hot_encoder",
        #     OneHotEncoder(
        #         variables=cat_var,
        #         drop_last=True
        #     ),
        # ),
        # perform binary encoding
        (
            "binary_encoder",
            ColumnTransformer(
                transformers=[
                    ("BE", pipe_encoder, cat_var),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # # perform Ordinal or numerical encoding
        # (
        #     "ordinal_encoder",
        #     OrdinalEncoder(
        #         encoding_method="ordered",
        #         variables=cat_var
        #     ),
        # ),
        # # perform count encoding
        # (
        #     "count_encoder",
        #     CountFrequencyEncoder(
        #         encoding_method="count",
        #         variables=cat_var
        #     ),
        # ),
        # # perform frequency encoding
        # (
        #     "frequency_encoder",
        #     CountFrequencyEncoder(
        #         encoding_method="frequency",
        #         variables=cat_var
        #     ),
        # ),
        # # perform mean encoding
        # (
        #     "mean_encoder",
        #     MeanEncoder(
        #         variables=cat_var
        #     ),
        # ),
        # # perform mean encoding
        # (
        #     "weight_of_evidence_encoder",
        #     WoEEncoder(
        #         variables=cat_var
        #     ),
        # ),
    ]
)


# ===== BUILDING PIPELINES TO SCALING DATA =====

# RobustScaler and QuantileTransformer are robust to outliers
# in the sense that adding or removing outliers in the training set
# will yield approximately the same transformation.
# But contrary to RobustScaler, QuantileTransformer will also automatically collapse any outlier
# by setting them to the a priori defined range boundaries (0 and 1).
# This can result in saturation artifacts for extreme values.

# all Scaler pipeline
# Standardizer for standard normalization
pipe_scaler_RobustScaler_scaler = Pipeline(
    steps=[
        ("RobustScaler_scaler", RobustScaler()),
    ]
)

pipe_StandardScaler_scaler = Pipeline(
    steps=[
        ("StandardScaler_scaler", StandardScaler()),
    ]
)

# Normalizer for normalization
pipe_MinMaxScaler_scaler = Pipeline(
    steps=[
        ("MinMaxScaler_scaler", MinMaxScaler()),
    ]
)

# pipe_minmax_scale_scaler = Pipeline(
#     steps=[
#         (
#             "minmax_scale_scaler",
#             minmax_scale()
#         ),
#     ]
# )

pipe_Normalizer_scaler = Pipeline(
    steps=[
        ("Normalizer_scaler", Normalizer()),
    ]
)

# Build pipeline
pipe_scaling_data_for_certain_model = Pipeline(
    [
        # to scale data using RobustScaler scaler
        (
            "RobustScaler_scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "RS",
                        pipe_scaler_RobustScaler_scaler,
                        make_column_selector(dtype_exclude=object),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # # to scale data using RobustScaler scaler
        # (
        #     "RobustScaler_scaler",
        #     RobustScaler(),
        # ),
        # # to scale data using standardization scaler
        # (
        #     "standardization_scaler",
        #     StandardScaler(),
        # ),
        # # to scale data using MinMaxScaler scaler
        # (
        #     "MinMaxScaler_scaler",
        #     MinMaxScaler(),
        # ),
        # # to scale data using minmax_scale scaler
        # (
        #     "minmax_scale_scaler",
        #     minmax_scale()
        # ),
        # # to scale data using Normalizer scaler
        # (
        #     "Normalizer_scaler",
        #     Normalizer(),
        # ),
    ]
)


# ===== BUILDING PIPELINES TO GENERATE NEW DATA SET USING NLP TRANSFORMERS =====
pipe_embed_data_with_NLP_transformers = Pipeline(
    [
        # # Transform data by USE transformers, developped by Google
        # (
        #     "USE_embedding_transformers",
        #     h_pp.GenerateEmbeddingDataWithUSETransformersTransform(
        #         module_url="https://tfhub.dev/google/universal-sentence-encoder-large/5"
        #     ),
        # ),
        # # Transform data by USE transformers, developped by Facebook
        # (
        #     "InferSent_embedding_transformers",
        #     h_pp.GenerateEmbeddingDataWithInferSentTransformersTransform(
        #         model_name_address="all-MiniLM-L6-v2"
        #     ),
        # ),
        # # Transform data by USE transformers, developped by Facebook
        # (
        #     "InferSent_embedding_transformers",
        #     h_pp.GenerateEmbeddingDataWithInferSentTransformersTransform(
        #         model_name_address="sentence-transformers/paraphrase-MiniLM-L6-v2"
        #     ),
        # ),
        # Transform data by Doc2Vec transformers
        (
            "Doc2Vec_embedding_transformers",
            h_pp.GenerateEmbeddingDataWithDoc2VecTransformersTransform(
                vector_size=100, window=5, min_count=1, workers=4, epochs=20
            ),
        ),
        # # Transform data by LASER transformers
        # (
        #     "LASER_embedding_transformers",
        #     h_pp.GenerateEmbeddingDataWithLASERTransformersTransform(),
        # ),
    ]
)


# ===== BUILDING PIPELINES TO DETECT MULTIVARIATE INLIERS AND OUTLIERS USING PyOD LIBRARY =====
# initialized a group of outlier detectors for acceleration
detector_list = [
    LOF(contamination=outlier_fraction),
    KNN(contamination=outlier_fraction),
    ECOD(contamination=outlier_fraction),
    OCSVM(contamination=outlier_fraction),
    COPOD(contamination=outlier_fraction),
    IForest(contamination=outlier_fraction),
    FeatureBagging(contamination=outlier_fraction, check_estimator=False),
]

# building pipeline
pipe_detect_multivariate_inliers_outliers_by_PyOD = Pipeline(
    [
        # Detect inliers and outliers using KNN detector
        (
            "KNN_detector",
            h_pp.DetecteMultivariateOutliersWithPyODTransform(
                model_name="KNN", model=KNN(contamination=outlier_fraction)
            ),
        ),
        # # Detect inliers and outliers using OneClassSVM detector
        # (
        #     "OneClassSVM_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="OneClassSVM", model=OCSVM(contamination=outlier_fraction)
        #     ),
        # ),
        # # Detect inliers and outliers using ECOD detector
        # (
        #     "ECOD_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="ECOD", model=ECOD(contamination=outlier_fraction)
        #     ),
        # ),
        # # Detect inliers and outliers using IForest detector
        # (
        #     "IForest_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="IForest", model=IForest(contamination=outlier_fraction)
        #     ),
        # ),
        # # Detect inliers and outliers using FeatureBagging detector
        # (
        #     "FeatureBagging_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="FeatureBagging",
        #         model=FeatureBagging(
        #             contamination=outlier_fraction, check_estimator=False
        #         ),
        #     ),
        # ),
        # # Detect inliers and outliers using COPOD detector
        # (
        #     "COPOD_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="COPOD", model=COPOD(contamination=outlier_fraction)
        #     ),
        # ),
        # # Detect inliers and outliers using SUOD detector
        # (
        #     "SUOD_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="SUOD",
        #         model=SUOD(
        #             base_estimators=detector_list,
        #             n_jobs=2,
        #             combination="average",
        #             verbose=False,
        #         ),
        #     ),
        # ),
        # # Detect inliers and outliers using LOF detector
        # (
        #     "LOF_detector",
        #     h_pp.DetecteMultivariateOutliersWithPyODTransform(
        #         model_name="LOF", model=LOF(contamination=outlier_fraction)
        #     ),
        # ),
    ]
)


# ===== BUILDING FEATURE SELECTION PIPELINES =====
# Define constante for pipeline method
const_tol = 1
quasi_const_tol = 0.99

# building pipeline
pipe_drop_unnecessary_feature = Pipeline(
    [
        # # to drop duplicated variable
        # (
        #     "drop_duplicated_variable_detected_with_pandas",
        #     DropFeatures(features_to_drop=duplicate_feature),
        # ),
        # to drop categorical variable with unique values = 1
        (
            "drop_categorical_variable_with_one_unique_value",
            DropFeatures(features_to_drop=constante_cat_feature),
        ),
        # # to drop numerical variable with std = 0
        # (
        #     "drop_numerical_variablewith_std_equal_to_0",
        #     DropFeatures(features_to_drop=constante_num_feature),
        # ),
        # Drop irrelevant features
        (
            "drop_irrelevant_feature",
            DropFeatures(features_to_drop=irrelevant_features),
        ),
    ]
)

# building pipeline
pipe_drop_feature_with_high_multicolinearity = Pipeline(
    [
        # to drop variable with high multicolinearity
        (
            "drop_variable_with_high_multicolinearity",
            DropFeatures(features_to_drop=num_var_with_high_multicolinearity),
        ),
    ]
)

# Drop fetaures from feature selection step
# building pipeline
pipe_drop_const_quasiconst_duplicated_feature = Pipeline(
    [
        # Detect and drop duplicated or redundant features
        (
            "drop_duplicated_feature",
            DropDuplicateFeatures(variables=None, missing_values="raise"),
        ),
        # Detect and drop constant feature
        (
            "drop_constante_feature",
            DropConstantFeatures(tol=const_tol, variables=None, missing_values="raise"),
        ),
        # # Detect and drop quasi constante feature
        # (
        #     "drop_quasi_constante_feature",
        #     DropConstantFeatures(
        #         tol=quasi_const_tol, variables=None, missing_values='raise'
        #     ),
        # ),
    ]
)

# Select fetaures comes from feature selection step
# building pipeline
pipe_select_feature_from_feature_selection = Pipeline(
    [
        # Select feature needed
        (
            "select_important_features",
            h_pp.SelectFeatureTransform(
                selected_variables=final_feature_selected_byLogReg
            ),
        ),
    ]
)

# Create pipeline for Logistic Regression model to use
# building pipeline
pipe_bagging_logistic_regression_model_default = Pipeline(
    [
        # Select the model needed
        (
            "bagging_logistic_regression_model_default",
            bagg_logreg_model_default,
        ),
    ]
)
pipe_bagging_logistic_regression_model_tuned = Pipeline(
    [
        # Select the model needed
        (
            "bagging_logistic_regression_model_tuned",
            bagg_logreg_model_tuned,
        ),
    ]
)

# Create pipeline for the ML model we will use for Interpretability step
# Define the ML model (Choose the model tuned by Optuna)
# # building pipeline example
# estimator = bagg_logreg_model_tuned
# pipe_ML_model_for_interpretability_step = Pipeline(
#     [
#         # Choose the model tuned by optuna here
#         (
#             "ML_model_tuned",
#             estimator,
#         ),
#     ]
# )
pipe_ML_model_for_interpretability_step = pipe_bagging_logistic_regression_model_tuned


# ===== BUILDING STACKING PIPELINES =====
# building pipeline
pipe_inhouse_stacking_estimator = Pipeline(
    [
        # to make first prediction using transformed data as input
        (
            "stacking_first_stage",
            h_pp.StackingEstimatorFirstStageTransform(
                variables=estimators_variables_selected,
                estimators=estimators,
                estimators_names=estimators_names,
                estimators_thresholds=estimators_thresholds,
            ),
        ),
        # # to make final prediction using first prediction output as input
        # (
        #     "stacking_final_stage",
        #     h_pp.StackingEstimatorSecondStageTransform(
        #         final_estimator=final_estimators,
        #         final_estimators_names=final_estimators_names,
        #         final_estimators_thresholds=final_estimators_thresholds
        #     ),
        # ),
    ]
)


# ===== BUILDING FINAL PIPELINES FOR PRODUCTION PREDICTION =====
# feature to use on frontend app in production
real_frontend_num_feature = ["pclass", "age", "sibsp", "parch", "fare"]
real_frontend_cat_feature = [
    "title",
    "sex",
    "ticket",
    "cabin",
    "embarked",
    "home",
    "destination",
]

# set the pipeline for first stage of stacking
stacking_first_stage_pipeline = Pipeline(
    [
        # to replace all missing values in data by np.nan
        (
            "replace_missing_value_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(missing_values_list=miss_val),
        ),
        # to cast all original variable into right type
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=real_frontend_num_feature,
                categorical_variable_list=real_frontend_cat_feature,
            ),
        ),
        # to extract all substring contains in the variable "ticket" in data set
        (
            "extract_substring_from_variable_ticket",
            h_pp_data_prep_clean.ExtractionSubstringFromTicketTransform(),
        ),
        # to extract all substring contains in the variable "cabin" in data set
        (
            "extract_substring_from_variable_cabin",
            h_pp_data_prep_clean.ExtractionSubstringFromCabinTransform(),
        ),
        # for drop some unneccesary variable
        # to drop categorical variable with unique values = 1
        (
            "drop_categorical_variable_with_one_unique_value",
            DropFeatures(features_to_drop=constante_cat_feature),
        ),
        # for somme, product and ratio variable generation
        (
            "somme_variable_generation",
            h_pp.GenerateSommeVariableTransform(variable_list=num_var_for_somme),
        ),
        (
            "product_variable_generation",
            h_pp.GenerateProductVariableTransform(variable_list=num_var_for_product),
        ),
        (
            "ratio_variable_generation",
            h_pp.GenerateRatioVariableTransform(variable_list=num_var_for_ratio),
        ),
        # for drop some unneccesary variable
        (
            "drop_variable_with_high_multicolinearity",
            DropFeatures(features_to_drop=num_var_with_high_multicolinearity),
        ),
        # for numerical variable MCAR imputation
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_num_var, random_state=seed),
        ),
        # for categorical variable MCAR imputation
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
        # for feature transformation to improve linear models
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=skew_num_var),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_Iqr_outliers_indicator_for_capping",
            h_pp.AddIqrOutliersIndicatorTransform(variable_list=num_var, fold=1.5),
        ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=num_var_with_high_variation_for_iqr,
                capping_method="iqr",
                tail="both",
                fold=1.5,
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with normal distribution
        (
            "outliers_capping_for_normal_variable_with_gaussian_method",
            Winsorizer(
                variables=num_var_with_low_variation_for_iqr,
                capping_method="gaussian",
                tail="both",
                fold=1.5,
            ),
        ),
        # for encoding of categorical variable
        # Re-group rare labels into "Rare"
        (
            "regroup_rare_labels_into_Rare",
            RareLabelEncoder(
                tol=0.05,  # minimal percentage to be considered non-rare
                n_categories=2,  # minimal number of categories required to re-group rare categorie
                variables=cat_var_with_rare_labels,  # variables to re-group
            ),
        ),
        # perform binary encoding
        (
            "binary_encoder",
            ColumnTransformer(
                transformers=[
                    ("BE", pipe_encoder, cat_var),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for scaling all data transformed
        # to scale data using RobustScaler scaler
        (
            "RobustScaler_scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "RS",
                        pipe_scaler_RobustScaler_scaler,
                        make_column_selector(dtype_exclude=object),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for detecting multivariate inliers and outliers
        # Detect inliers and outliers using KNN detector
        (
            "KNN_detector",
            h_pp.DetecteMultivariateOutliersWithPyODTransform(
                model_name="KNN", model=KNN(contamination=outlier_fraction)
            ),
        ),
        # to drop constante, quasi-constante and redundant features in data set
        # Detect and drop duplicated or redundant features
        (
            "drop_duplicated_feature",
            DropDuplicateFeatures(variables=None, missing_values="raise"),
        ),
        # Detect and drop constant feature
        (
            "drop_constante_feature",
            DropConstantFeatures(tol=const_tol, variables=None, missing_values="raise"),
        ),
    ]
)


# set the pipeline for second stage of stacking
stacking_second_stage_pipeline = Pipeline(
    [
        # to replace all missing values in data by np.nan
        (
            "replace_missing_value_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(missing_values_list=miss_val),
        ),
        # to cast all original variable into right type
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=real_frontend_num_feature,
                categorical_variable_list=real_frontend_cat_feature,
            ),
        ),
        # to extract all substring contains in the variable "ticket" in data set
        (
            "extract_substring_from_variable_ticket",
            h_pp_data_prep_clean.ExtractionSubstringFromTicketTransform(),
        ),
        # to extract all substring contains in the variable "cabin" in data set
        (
            "extract_substring_from_variable_cabin",
            h_pp_data_prep_clean.ExtractionSubstringFromCabinTransform(),
        ),
        # for drop some unneccesary variable
        # to drop categorical variable with unique values = 1
        (
            "drop_categorical_variable_with_one_unique_value",
            DropFeatures(features_to_drop=constante_cat_feature),
        ),
        # for somme, product and ratio variable generation
        (
            "somme_variable_generation",
            h_pp.GenerateSommeVariableTransform(variable_list=num_var_for_somme),
        ),
        (
            "product_variable_generation",
            h_pp.GenerateProductVariableTransform(variable_list=num_var_for_product),
        ),
        (
            "ratio_variable_generation",
            h_pp.GenerateRatioVariableTransform(variable_list=num_var_for_ratio),
        ),
        # for drop some unneccesary variable
        (
            "drop_variable_with_high_multicolinearity",
            DropFeatures(features_to_drop=num_var_with_high_multicolinearity),
        ),
        # for numerical variable MCAR imputation
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_num_var, random_state=seed),
        ),
        # for categorical variable MCAR imputation
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
        # for feature transformation to improve linear models
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=skew_num_var),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_Iqr_outliers_indicator_for_capping",
            h_pp.AddIqrOutliersIndicatorTransform(variable_list=num_var, fold=1.5),
        ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=num_var_with_high_variation_for_iqr,
                capping_method="iqr",
                tail="both",
                fold=1.5,
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with normal distribution
        (
            "outliers_capping_for_normal_variable_with_gaussian_method",
            Winsorizer(
                variables=num_var_with_low_variation_for_iqr,
                capping_method="gaussian",
                tail="both",
                fold=1.5,
            ),
        ),
        # for encoding of categorical variable
        # Re-group rare labels into "Rare"
        (
            "regroup_rare_labels_into_Rare",
            RareLabelEncoder(
                tol=0.05,  # minimal percentage to be considered non-rare
                n_categories=2,  # minimal number of categories required to re-group rare categorie
                variables=cat_var_with_rare_labels,  # variables to re-group
            ),
        ),
        # perform binary encoding
        (
            "binary_encoder",
            ColumnTransformer(
                transformers=[
                    ("BE", pipe_encoder, cat_var),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for scaling all data transformed
        # to scale data using RobustScaler scaler
        (
            "RobustScaler_scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "RS",
                        pipe_scaler_RobustScaler_scaler,
                        make_column_selector(dtype_exclude=object),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for detecting multivariate inliers and outliers
        # Detect inliers and outliers using KNN detector
        (
            "KNN_detector",
            h_pp.DetecteMultivariateOutliersWithPyODTransform(
                model_name="KNN", model=KNN(contamination=outlier_fraction)
            ),
        ),
        # to drop constante, quasi-constante and redundant features in data set
        # Detect and drop duplicated or redundant features
        (
            "drop_duplicated_feature",
            DropDuplicateFeatures(variables=None, missing_values="raise"),
        ),
        # Detect and drop constant feature
        (
            "drop_constante_feature",
            DropConstantFeatures(tol=const_tol, variables=None, missing_values="raise"),
        ),
        # to make a prediction with inhouse stacking estimator
        # to make first prediction using transformed data as input
        (
            "stacking_first_stage",
            h_pp.StackingEstimatorFirstStageTransform(
                variables=estimators_variables_selected,
                estimators=estimators,
                estimators_names=estimators_names,
                estimators_thresholds=estimators_thresholds,
            ),
        ),
    ]
)


# set the final pipeline for production prediction
production_prediction_pipeline = Pipeline(
    [
        # to replace all missing values in data by np.nan
        (
            "replace_missing_value_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(missing_values_list=miss_val),
        ),
        # to cast all original variable into right type
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=real_frontend_num_feature,
                categorical_variable_list=real_frontend_cat_feature,
            ),
        ),
        # to extract all substring contains in the variable "ticket" in data set
        (
            "extract_substring_from_variable_ticket",
            h_pp_data_prep_clean.ExtractionSubstringFromTicketTransform(),
        ),
        # to extract all substring contains in the variable "cabin" in data set
        (
            "extract_substring_from_variable_cabin",
            h_pp_data_prep_clean.ExtractionSubstringFromCabinTransform(),
        ),
        # for drop some unneccesary variable
        # to drop categorical variable with unique values = 1
        (
            "drop_categorical_variable_with_one_unique_value",
            DropFeatures(features_to_drop=constante_cat_feature),
        ),
        # for somme, product and ratio variable generation
        (
            "somme_variable_generation",
            h_pp.GenerateSommeVariableTransform(variable_list=num_var_for_somme),
        ),
        (
            "product_variable_generation",
            h_pp.GenerateProductVariableTransform(variable_list=num_var_for_product),
        ),
        (
            "ratio_variable_generation",
            h_pp.GenerateRatioVariableTransform(variable_list=num_var_for_ratio),
        ),
        # for drop some unneccesary variable
        (
            "drop_variable_with_high_multicolinearity",
            DropFeatures(features_to_drop=num_var_with_high_multicolinearity),
        ),
        # for numerical variable MCAR imputation
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_num_var, random_state=seed),
        ),
        # for categorical variable MCAR imputation
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
        # for feature transformation to improve linear models
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=skew_num_var),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_Iqr_outliers_indicator_for_capping",
            h_pp.AddIqrOutliersIndicatorTransform(variable_list=num_var, fold=1.5),
        ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=num_var_with_high_variation_for_iqr,
                capping_method="iqr",
                tail="both",
                fold=1.5,
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with normal distribution
        (
            "outliers_capping_for_normal_variable_with_gaussian_method",
            Winsorizer(
                variables=num_var_with_low_variation_for_iqr,
                capping_method="gaussian",
                tail="both",
                fold=1.5,
            ),
        ),
        # for encoding of categorical variable
        # Re-group rare labels into "Rare"
        (
            "regroup_rare_labels_into_Rare",
            RareLabelEncoder(
                tol=0.05,  # minimal percentage to be considered non-rare
                n_categories=2,  # minimal number of categories required to re-group rare categorie
                variables=cat_var_with_rare_labels,  # variables to re-group
            ),
        ),
        # perform binary encoding
        (
            "binary_encoder",
            ColumnTransformer(
                transformers=[
                    ("BE", pipe_encoder, cat_var),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for scaling all data transformed
        # to scale data using RobustScaler scaler
        (
            "RobustScaler_scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "RS",
                        pipe_scaler_RobustScaler_scaler,
                        make_column_selector(dtype_exclude=object),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for detecting multivariate inliers and outliers
        # Detect inliers and outliers using KNN detector
        (
            "KNN_detector",
            h_pp.DetecteMultivariateOutliersWithPyODTransform(
                model_name="KNN", model=KNN(contamination=outlier_fraction)
            ),
        ),
        # to drop constante, quasi-constante and redundant features in data set
        # Detect and drop duplicated or redundant features
        (
            "drop_duplicated_feature",
            DropDuplicateFeatures(variables=None, missing_values="raise"),
        ),
        # Detect and drop constant feature
        (
            "drop_constante_feature",
            DropConstantFeatures(tol=const_tol, variables=None, missing_values="raise"),
        ),
        # to make a prediction with inhouse stacking estimator
        # to make first prediction using transformed data as input
        (
            "stacking_first_stage",
            h_pp.StackingEstimatorFirstStageTransform(
                variables=estimators_variables_selected,
                estimators=estimators,
                estimators_names=estimators_names,
                estimators_thresholds=estimators_thresholds,
            ),
        ),
        # to make final prediction using first prediction output as input
        (
            "final_estimator",
            h_pp.FinalEstimatorTransform(
                variables=final_estimators_variables_selected,
                estimators=final_estimators,
                estimators_names=final_estimators_names,
                estimators_thresholds=final_estimators_thresholds,
            ),
        ),
    ]
)


# set the final pipeline for interpretability in production
interpretability_pipeline = Pipeline(
    [
        # to mapp input data into right way
        (
            "mapping_data_with_dictionnary",
            h_pp.Mapper(
                variables=real_frontend_cat_feature,
                mappings=numtocat_features_dict,
            ),
        ),
        # to replace all missing values in data by np.nan
        (
            "replace_missing_value_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(missing_values_list=miss_val),
        ),
        # to cast all original variable into right type
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=real_frontend_num_feature,
                categorical_variable_list=real_frontend_cat_feature,
            ),
        ),
        # to extract all substring contains in the variable "ticket" in data set
        (
            "extract_substring_from_variable_ticket",
            h_pp_data_prep_clean.ExtractionSubstringFromTicketTransform(),
        ),
        # to extract all substring contains in the variable "cabin" in data set
        (
            "extract_substring_from_variable_cabin",
            h_pp_data_prep_clean.ExtractionSubstringFromCabinTransform(),
        ),
        # for drop some unneccesary variable
        # to drop categorical variable with unique values = 1
        (
            "drop_categorical_variable_with_one_unique_value",
            DropFeatures(features_to_drop=constante_cat_feature),
        ),
        # for somme, product and ratio variable generation
        (
            "somme_variable_generation",
            h_pp.GenerateSommeVariableTransform(variable_list=num_var_for_somme),
        ),
        (
            "product_variable_generation",
            h_pp.GenerateProductVariableTransform(variable_list=num_var_for_product),
        ),
        (
            "ratio_variable_generation",
            h_pp.GenerateRatioVariableTransform(variable_list=num_var_for_ratio),
        ),
        # for drop some unneccesary variable
        (
            "drop_variable_with_high_multicolinearity",
            DropFeatures(features_to_drop=num_var_with_high_multicolinearity),
        ),
        # for numerical variable MCAR imputation
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_num_var, random_state=seed),
        ),
        # for categorical variable MCAR imputation
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
        # for feature transformation to improve linear models
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=skew_num_var),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_Iqr_outliers_indicator_for_capping",
            h_pp.AddIqrOutliersIndicatorTransform(variable_list=num_var, fold=1.5),
        ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=num_var_with_high_variation_for_iqr,
                capping_method="iqr",
                tail="both",
                fold=1.5,
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with normal distribution
        (
            "outliers_capping_for_normal_variable_with_gaussian_method",
            Winsorizer(
                variables=num_var_with_low_variation_for_iqr,
                capping_method="gaussian",
                tail="both",
                fold=1.5,
            ),
        ),
        # for encoding of categorical variable
        # Re-group rare labels into "Rare"
        (
            "regroup_rare_labels_into_Rare",
            RareLabelEncoder(
                tol=0.05,  # minimal percentage to be considered non-rare
                n_categories=2,  # minimal number of categories required to re-group rare categorie
                variables=cat_var_with_rare_labels,  # variables to re-group
            ),
        ),
        # perform binary encoding
        (
            "binary_encoder",
            ColumnTransformer(
                transformers=[
                    ("BE", pipe_encoder, cat_var),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for scaling all data transformed
        # to scale data using RobustScaler scaler
        (
            "RobustScaler_scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "RS",
                        pipe_scaler_RobustScaler_scaler,
                        make_column_selector(dtype_exclude=object),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for detecting multivariate inliers and outliers
        # Detect inliers and outliers using KNN detector
        (
            "KNN_detector",
            h_pp.DetecteMultivariateOutliersWithPyODTransform(
                model_name="KNN", model=KNN(contamination=outlier_fraction)
            ),
        ),
        # to drop constante, quasi-constante and redundant features in data set
        # Detect and drop duplicated or redundant features
        (
            "drop_duplicated_feature",
            DropDuplicateFeatures(variables=None, missing_values="raise"),
        ),
        # Detect and drop constant feature
        (
            "drop_constante_feature",
            DropConstantFeatures(tol=const_tol, variables=None, missing_values="raise"),
        ),
        # to make a prediction with inhouse stacking estimator
        # to make first prediction using transformed data as input
        (
            "stacking_first_stage",
            h_pp.StackingEstimatorFirstStageTransform(
                variables=estimators_variables_selected,
                estimators=estimators,
                estimators_names=estimators_names,
                estimators_thresholds=estimators_thresholds,
            ),
        ),
        # to make final prediction using first prediction output as input
        (
            "final_estimator",
            h_pp.FinalEstimatorTransform(
                variables=final_estimators_variables_selected,
                estimators=final_estimators,
                estimators_names=final_estimators_names,
                estimators_thresholds=final_estimators_thresholds,
            ),
        ),
    ]
)


# set pipeline for mapping categorical feature unique value
num_to_cat_mapper_pipeline = Pipeline(
    [
        # to mapp input data into right way
        (
            "mapping_data_with_dictionnary",
            h_pp.Mapper(
                variables=real_frontend_cat_feature,
                mappings=numtocat_features_dict,
            ),
        ),
    ]
)

cat_to_num_mapper_pipeline = Pipeline(
    [
        # to mapp input data into right way
        (
            "mapping_data_with_dictionnary",
            h_pp.Mapper(
                variables=real_frontend_cat_feature,
                mappings=cattonum_features_dict,
            ),
        ),
    ]
)


# set the final pipeline for production prediction
production_prediction_pipeline_old = Pipeline(
    [
        # to replace all missing values in data by np.nan
        (
            "replace_missing_value_by_NAN",
            h_pp.ReplaceMissingValueByNanTransform(missing_values_list=miss_val),
        ),
        # to cast all original variable into right type
        (
            "assign_right_type",
            h_pp.AssignRightTypeToAllVariableTransform(
                numerical_variable_list=real_frontend_num_feature,
                categorical_variable_list=real_frontend_cat_feature,
            ),
        ),
        # # for checking and rename the name of column "home.dest" to "homedest"
        # (
        #     "check_rename_certain_column_name",
        #     h_pp_data_prep_clean.CheckAndRenameHomedestColumnNameTransform(),
        # ),
        # # to extract all title contains in the variable "name" in data set
        # (
        #     "extract_all_title_in_variable_name",
        #     h_pp_data_prep_clean.ExtractionAllTitleFromTheNameTransform(),
        # ),
        # to extract all substring contains in the variable "ticket" in data set
        (
            "extract_substring_from_variable_ticket",
            h_pp_data_prep_clean.ExtractionSubstringFromTicketTransform(),
        ),
        # to extract all substring contains in the variable "cabin" in data set
        (
            "extract_substring_from_variable_cabin",
            h_pp_data_prep_clean.ExtractionSubstringFromCabinTransform(),
        ),
        # # to extract all substring contains in the variable "homedest" in data set
        # (
        #     "extract_substring_from_variable_homedest",
        #     h_pp_data_prep_clean.ExtractionSubstringFromHomeDestTransform(),
        # ),
        # for drop some unneccesary variable
        # to drop categorical variable with unique values = 1
        (
            "drop_categorical_variable_with_one_unique_value",
            DropFeatures(features_to_drop=constante_cat_feature),
        ),
        # # Drop irrelevant features
        # (
        #     "drop_irrelevant_feature",
        #     DropFeatures(features_to_drop=irrelevant_features),
        # ),
        # # to cast all variable into right type
        # (
        #     "assign_right_type",
        #     h_pp.AssignRightTypeToAllVariableTransform(
        #         numerical_variable_list=num_var_prep_clean_data,
        #         categorical_variable_list=cat_var_prep_clean_data,
        #     ),
        # ),
        # for somme, product and ratio variable generation
        (
            "somme_variable_generation",
            h_pp.GenerateSommeVariableTransform(variable_list=num_var_for_somme),
        ),
        (
            "product_variable_generation",
            h_pp.GenerateProductVariableTransform(variable_list=num_var_for_product),
        ),
        (
            "ratio_variable_generation",
            h_pp.GenerateRatioVariableTransform(variable_list=num_var_for_ratio),
        ),
        # for drop some unneccesary variable
        (
            "drop_variable_with_high_multicolinearity",
            DropFeatures(features_to_drop=num_var_with_high_multicolinearity),
        ),
        # for numerical variable MCAR imputation
        (
            "random_sample_imputation",
            RandomSampleImputer(variables=miss_num_var, random_state=seed),
        ),
        # for categorical variable MCAR imputation
        (
            "frequent_category_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=miss_cat_var,
            ),
        ),
        # for feature transformation to improve linear models
        # Convert variable with skewed distribution into normal distribution
        # to apply quantile transformation to continuous and discrete variable
        (
            "quantile_transformation_application",
            h_pp.QuantileTransformerTransform(variable_list=skew_num_var),
        ),
        # Outliers handling
        # Add Outliers indicator
        (
            "add_Iqr_outliers_indicator_for_capping",
            h_pp.AddIqrOutliersIndicatorTransform(variable_list=num_var, fold=1.5),
        ),
        # Outliers handling
        # to cap outliers in the variable with skewed distribution
        (
            "outliers_capping_for_skewed_variable_with_IQR_method",
            Winsorizer(
                variables=num_var_with_high_variation_for_iqr,
                capping_method="iqr",
                tail="both",
                fold=1.5,
            ),
        ),
        # Outliers handling
        # to cap outliers in the variable with normal distribution
        (
            "outliers_capping_for_normal_variable_with_gaussian_method",
            Winsorizer(
                variables=num_var_with_low_variation_for_iqr,
                capping_method="gaussian",
                tail="both",
                fold=1.5,
            ),
        ),
        # for encoding of categorical variable
        # Re-group rare labels into "Rare"
        (
            "regroup_rare_labels_into_Rare",
            RareLabelEncoder(
                tol=0.05,  # minimal percentage to be considered non-rare
                n_categories=2,  # minimal number of categories required to re-group rare categorie
                variables=cat_var_with_rare_labels,  # variables to re-group
            ),
        ),
        # perform binary encoding
        (
            "binary_encoder",
            ColumnTransformer(
                transformers=[
                    ("BE", pipe_encoder, cat_var),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for scaling all data transformed
        # to scale data using RobustScaler scaler
        (
            "RobustScaler_scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "RS",
                        pipe_scaler_RobustScaler_scaler,
                        make_column_selector(dtype_exclude=object),
                    ),
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        # for detecting multivariate inliers and outliers
        # Detect inliers and outliers using KNN detector
        (
            "KNN_detector",
            h_pp.DetecteMultivariateOutliersWithPyODTransform(
                model_name="KNN", model=KNN(contamination=outlier_fraction)
            ),
        ),
        # to drop constante, quasi-constante and redundant features in data set
        # Detect and drop duplicated or redundant features
        (
            "drop_duplicated_feature",
            DropDuplicateFeatures(variables=None, missing_values="raise"),
        ),
        # Detect and drop constant feature
        (
            "drop_constante_feature",
            DropConstantFeatures(tol=const_tol, variables=None, missing_values="raise"),
        ),
        # to make a prediction with inhouse stacking estimator
        # to make first prediction using transformed data as input
        (
            "stacking_first_stage",
            h_pp.StackingEstimatorFirstStageTransform(
                variables=estimators_variables_selected,
                estimators=estimators,
                estimators_names=estimators_names,
                estimators_thresholds=estimators_thresholds,
            ),
        ),
        # # to make final prediction using first prediction output as input
        # (
        #     "stacking_final_stage",
        #     h_pp.StackingEstimatorSecondStageTransform(
        #         final_estimator=final_estimators,
        #         final_estimators_names=final_estimators_names,
        #         final_estimators_thresholds=final_estimators_thresholds
        #     ),
        # ),
        # # to make final prediction using first prediction output as input
        # (
        #     "final_estimator",
        #     RandomForestClassifier(
        #         max_depth=2, random_state=seed
        #     ),
        # ),
        # to make final prediction using first prediction output as input
        (
            "final_estimator",
            h_pp.FinalEstimatorTransform(
                variables=final_estimators_variables_selected,
                estimators=final_estimators,
                estimators_names=final_estimators_names,
                estimators_thresholds=final_estimators_thresholds,
            ),
        ),
    ]
)
