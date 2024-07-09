import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib

# Read the parquet file using Dask
dft = dd.read_parquet("/home/samanehjavadinia/Co-op/Data/cfht_ngvs_lite_simbad.parquet")

# Set display options to show all columns
pd.set_option("display.max_columns", None)

# Print all column names
print(list(dft.columns))

# Define the features you want to keep
features = [
    "A_WORLD",
    "B_WORLD",
    "THETA_WORLD",
    "ERRA_WORLD",
    "ERRB_WORLD",
    "ERRTHETA_WORLD",
    "EBV",
    "U_MAG_ISO",
    "G_MAG_ISO",
    "R_MAG_ISO",
    "I_MAG_ISO",
    "Z_MAG_ISO",
    "U_MAGERR_ISO",
    "G_MAGERR_ISO",
    "R_MAGERR_ISO",
    "I_MAGERR_ISO",
    "Z_MAGERR_ISO",
    "U_MAG_APER1",
    "G_MAG_APER1",
    "R_MAG_APER1",
    "I_MAG_APER1",
    "Z_MAG_APER1",
    "U_MAG_APER3",
    "G_MAG_APER3",
    "R_MAG_APER3",
    "I_MAG_APER3",
    "Z_MAG_APER3",
    "U_MAG_APER5",
    "G_MAG_APER5",
    "R_MAG_APER5",
    "I_MAG_APER5",
    "Z_MAG_APER5",
    "U_MAGERR_APER1",
    "G_MAGERR_APER1",
    "R_MAGERR_APER1",
    "I_MAGERR_APER1",
    "Z_MAGERR_APER1",
    "U_MAGERR_APER3",
    "G_MAGERR_APER3",
    "R_MAGERR_APER3",
    "I_MAGERR_APER3",
    "Z_MAGERR_APER3",
    "U_MAGERR_APER5",
    "G_MAGERR_APER5",
    "R_MAGERR_APER5",
    "I_MAGERR_APER5",
    "Z_MAGERR_APER5",
    "U_MU_MAX",
    "G_MU_MAX",
    "R_MU_MAX",
    "I_MU_MAX",
    "Z_MU_MAX",
    "U_BACKGROUND",
    "G_BACKGROUND",
    "R_BACKGROUND",
    "I_BACKGROUND",
    "Z_BACKGROUND",
    "U_ISOAREA_IMAGE",
    "G_ISOAREA_IMAGE",
    "R_ISOAREA_IMAGE",
    "I_ISOAREA_IMAGE",
    "Z_ISOAREA_IMAGE",
    "U_FWHM_IMAGE",
    "G_FWHM_IMAGE",
    "R_FWHM_IMAGE",
    "I_FWHM_IMAGE",
    "Z_FWHM_IMAGE",
    "U_FLUX_RADIUS",
    "G_FLUX_RADIUS",
    "R_FLUX_RADIUS",
    "I_FLUX_RADIUS",
    "Z_FLUX_RADIUS",
    "U_KRON_RADIUS",
    "G_KRON_RADIUS",
    "R_KRON_RADIUS",
    "I_KRON_RADIUS",
    "Z_KRON_RADIUS",
    "U_PETRO_RADIUS",
    "G_PETRO_RADIUS",
    "R_PETRO_RADIUS",
    "I_PETRO_RADIUS",
    "Z_PETRO_RADIUS",
    "main_type",
]

# Select only the specified features
dft = dft[features]

# Drop rows where 'main_type' is NaN
dft_cleaned = dft.dropna(subset=["main_type"])

# Separate unlabeled data
unlabeled_data = dft[dft["main_type"].isna()]


# # Define function to drop rows with value 99
# def drop_rows_with_99(df):
#     return df[~(df == 99).any(axis=1)]


# # Apply the drop_rows_with_99 function
# dft_cleaned = drop_rows_with_99(dft_cleaned)
# unlabeled_data_cleaned = drop_rows_with_99(unlabeled_data)

# # Filter to omit rows where umag is more than 30 or err is more than 0.5
# filtered_dft = dft_cleaned[
#     (dft_cleaned["U_MAG_ISO"] <= 30)
#     & (dft_cleaned["U_MAGERR_ISO"] <= 0.5)
#     & (dft_cleaned["G_MAGERR_ISO"] <= 0.5)
#     & (dft_cleaned["R_MAGERR_ISO"] <= 0.5)
#     & (dft_cleaned["I_MAGERR_ISO"] <= 0.5)
# ]

# filtered_unlabeled_data = unlabeled_data_cleaned[
#     (unlabeled_data_cleaned["U_MAG_ISO"] <= 30)
#     & (unlabeled_data_cleaned["U_MAGERR_ISO"] <= 0.5)
#     & (unlabeled_data_cleaned["G_MAGERR_ISO"] <= 0.5)
#     & (unlabeled_data_cleaned["R_MAGERR_ISO"] <= 0.5)
#     & (unlabeled_data_cleaned["I_MAGERR_ISO"] <= 0.5)
# ]


# # Create function to print value counts of columns with fewer than 5 unique values
# def print_columns_with_fewer_than_5_unique_values(df):
#     for column in df.columns:
#         unique_values_count = df[column].nunique()
#         if unique_values_count < 5:
#             print(
#                 f"Value counts for column '{column}' (unique values: {unique_values_count}):"
#             )
#             print(df[column].value_counts())
#             print("\n")


# # Create function to print value counts of columns with fewer than 5 unique values
# def print_columns_with_fewer_than_5_unique_values(df):
#     for column in df.columns:
#         unique_values_count = df[column].nunique().compute()
#         if unique_values_count < 5:
#             print(
#                 f"Value counts for column '{column}' (unique values: {unique_values_count}):"
#             )
#             value_counts = df[column].value_counts().compute()
#             print(value_counts)
#             print("\n")


# # Print columns with fewer than 5 unique values
# print_columns_with_fewer_than_5_unique_values(filtered_dft)
# print_columns_with_fewer_than_5_unique_values(filtered_unlabeled_data)


# # Define function to create feature differences and error combinations
# def create_feature_differences_with_errors(
#     df, feature_groups, error_groups, proxy_feature
# ):
#     for group, error_group in zip(feature_groups, error_groups):
#         for i in range(len(group)):
#             for j in range(i + 1, len(group)):
#                 diff_feature_name = f"{group[i]}_minus_{group[j]}"
#                 error_feature_name = f"{error_group[i]}_minus_{error_group[j]}"

#                 df[diff_feature_name] = df[group[i]] - df[group[j]]
#                 df[error_feature_name] = np.sqrt(
#                     df[error_group[i]] ** 2 + df[error_group[j]] ** 2
#                 )

#         features_to_drop = [feature for feature in group if feature != proxy_feature]
#         df.drop(columns=features_to_drop, inplace=True)
#         df.drop(columns=error_group, inplace=True)

#     return df


# # Define the proxy feature to retain
# proxy_feature = "G_MAG_ISO"

# # Define the feature groups and corresponding error groups
# feature_groups = [["U_MAG_ISO", "G_MAG_ISO", "R_MAG_ISO", "I_MAG_ISO", "Z_MAG_ISO"]]
# error_groups = [
#     ["U_MAGERR_ISO", "G_MAGERR_ISO", "R_MAGERR_ISO", "I_MAGERR_ISO", "Z_MAGERR_ISO"]
# ]

# # Apply the function
# dft = create_feature_differences_with_errors(
#     filtered_dft, feature_groups, error_groups, proxy_feature
# )
# unlabeled_data = create_feature_differences_with_errors(
#     filtered_unlabeled_data, feature_groups, error_groups, proxy_feature
# )

# # Group categories in 'main_type'
# threshold = 300
# category_counts = dft["main_type"].value_counts()
# rare_categories = category_counts[category_counts <= threshold].index
# dft["grouped_category"] = dft["main_type"].apply(
#     lambda x: "Other" if x in rare_categories else x
# )

# # Ensure all values in 'grouped_category' are strings
# dft["grouped_category"] = dft["grouped_category"].astype(str)

# # Label Encoding
# label_encoder = LabelEncoder()
# dft["main_type_encoded"] = label_encoder.fit_transform(dft["grouped_category"])

# # Save the encoder to disk
# joblib.dump(label_encoder, "label_encoder.pkl")

# # Calculate the correlation matrix
# correlation_matrix = dft.corr()
# correlation_with_target = correlation_matrix["main_type_encoded"].drop(
#     "main_type_encoded"
# )

# # Plot the correlations
# sorted_correlations = correlation_with_target.sort_values()
# plt.figure(figsize=(10, 6))
# sns.barplot(
#     x=sorted_correlations.values, y=sorted_correlations.index, palette="viridis"
# )
# plt.title("Feature Correlations with Target")
# plt.xlabel("Correlation Coefficient")
# plt.ylabel("Features")
# plt.show()


# # Drop unnecessary columns
# dft = dft.drop(columns=["main_type", "grouped_category"])
# unlabeled_data = unlabeled_data.drop(columns=["main_type"])

# Save the train and test datasets to CSV files
dft.to_csv("/home/samanehjavadinia/Co-op/Data/CFHT_modified_dataset.csv", index=False)
unlabeled_data.to_csv(
    "/home/samanehjavadinia/Co-op/Data/CFHT_modified_unlabeled_data.csv", index=False
)
