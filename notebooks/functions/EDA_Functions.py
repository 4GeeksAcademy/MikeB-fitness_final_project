'''Helper functions for EDA/cleaning of categorical variables.'''
import pandas as pd
import numpy as np

from typing import Callable
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


pd.set_option('display.max_columns', None)

'''Load Dataset'''
def load_data(file_path):
    '''Load dataset from a CSV File'''   
    return pd.read_csv(file_path)

'''Summary Statistics'''
def summarize_data(df):
    """Prints key statistics, detects missing values, and provides distribution insights."""
    
    # General summary statistics (same as before)
    print("\nBasic Summary Statistics:\n", df.describe(percentiles=[0.25, 0.50, 0.75]))

    # Check missing values
    print("\nMissing values:\n", df.isnull().sum())

    # Compute skewness & kurtosis for numeric columns
    skew_kurt_data = {
        feature: {"Skewness": skew(df[feature]), "Kurtosis": kurtosis(df[feature])}
        for feature in df.select_dtypes(include=['number']).columns
    }
    
    skew_kurt_df = pd.DataFrame(skew_kurt_data).T  # Convert dictionary to DataFrame
    
    print("\nDistribution Insights (Skewness & Kurtosis):\n", skew_kurt_df)
    
    return skew_kurt_df  # Return the distribution insights if needed elsewhere


'''Apply One-Hot Encoding'''
def hot_encode_categorical(df):
    """Performs One-Hot Encoding on categorical features."""
    
    categorical_features = ['Gender']
    
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    
    return df

def ordinal_encode_features(df, categorical_features):
    """
    Dynamically applies ordinal encoding to specified categorical columns.

    Args:
        df (pd.DataFrame): DataFrame containing categorical features to encode.
        categorical_features (list): List of categorical column names.

    Returns:
        pd.DataFrame: Encoded DataFrame with ordinal values.
    """
    # Extract unique values for each categorical feature
    categories_list = [df[col].unique().tolist() for col in categorical_features]

    # Apply dynamic ordinal encoding
    encoder = OrdinalEncoder(categories=categories_list)
    df_encoded = df.copy()  # Preserve original dataframe
    df_encoded[categorical_features] = encoder.fit_transform(df[categorical_features])

    return df_encoded

'''Efficiency Score Calculation'''
def calculate_efficiency(df):

    """Computes workout efficiency based on calorie burn rate and heart rate."""
    df['Efficiency_score'] = df['Calories'] / df['Duration'] * (df['Heart_Rate'] / 100)
    return df

def rank_feature_importance(df, target_feature):
    """Ranks features by their correlation strength with the target metric."""
    correlations = df.corr()[target_feature].drop(target_feature).abs().sort_values(ascending=False)
    return correlations.head(10)  # Show top 10 impactful features

def select_high_correlation_features(df, target_feature, threshold=0.5):
    """Filters dataset to only keep high-correlation features."""
    correlation_matrix = df.corr()
    high_corr_features = correlation_matrix[target_feature][correlation_matrix[target_feature].abs() > threshold].index.tolist()
    return df[high_corr_features]

def get_correlations(feature_pairs: list, df: pd.DataFrame, correlations: dict=None) -> dict:
    '''Takes list of feature name tuples and a dataframe, calculates Pearson 
    correlation coefficient and Spearman rank correlation coefficient using
    SciPy. Returns a dictionary with the results. Pass in results from an
    earlier call to append.'''

    # If results weren't passed in, start an empty dictionary to collect them
    if correlations is None:
        correlations={
            'Feature 1':[],
            'Feature 2':[],
            'Absolute Spearman':[],
            'Spearman':[],
            'Spearman p-value':[],
            'Absolute Pearson':[],
            'Pearson':[],
            'Pearson p-value':[],
            'Pearson r-squared':[]
        }

    # Loop on the feature pairs to calculate the corelation coefficients between each
    for feature_pair in feature_pairs:

        # Exclude self pairs
        if feature_pair[0] != feature_pair[1]:

            # Get data for this feature pair
            feature_pair_data=df[[*feature_pair]].copy()

            # Replace any infinite values with nan and drop
            feature_pair_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            feature_pair_data.dropna(inplace=True)

            # Get Pearson and Spearman correlation coefficients and their p-values
            pcc=pearsonr(feature_pair_data.iloc[:,0], feature_pair_data.iloc[:,1])
            src=spearmanr(feature_pair_data.iloc[:,0], feature_pair_data.iloc[:,1])

            # Collect the results
            correlations['Feature 1'].append(feature_pair[0])
            correlations['Feature 2'].append(feature_pair[1])
            correlations['Absolute Spearman'].append(abs(src.statistic))
            correlations['Spearman'].append(src.statistic)
            correlations['Spearman p-value'].append(src.pvalue)
            correlations['Absolute Pearson'].append(pcc.statistic)
            correlations['Pearson'].append(pcc.statistic)
            correlations['Pearson p-value'].append(pcc.pvalue)
            correlations['Pearson r-squared'].append(pcc.statistic**2)

    return correlations


def test_features(
        model: Callable,
        datasets: dict,
        label: str='price',
        scoring: str='explained_variance',
        folds: int=30,
) -> dict:
    '''Runs cross-validation on data in datasets dictionary.'''

    results={
        'Feature set':[],
        'Score':[]
    }

    for dataset, df in datasets.items():

        cleaned_df=df.dropna(inplace=False).copy()
        #print(f'{dataset}: {cleaned_df.columns}')

        scores=cross_val_score(
            model,
            cleaned_df.drop(label, axis=1),
            cleaned_df[label],
            scoring=scoring,
            cv=ShuffleSplit(n_splits=folds, test_size=0.25, random_state=315)
            
        )
        
        #print(f'Scores: {scores}')

        results['Feature set'].extend([dataset]*folds)
        results['Score'].extend(abs(scores))

    return pd.DataFrame.from_dict(results)


def evaluate_datasets(
        model: Callable,
        datasets: dict,
        label: str='price',
        scoring: str='explained_variance',
        folds: int=30
):
    '''Takes a Scikit-learn regression model instance, a dictionary of datasets, the label column name and
    a SciKit-Learn scoring string. Run cross-validation on each dataset, then tests the differences
    in scores with ANOVA followed by Tukey's post-hoc test. Returns cross-validation scores an Tukey result'''

    cross_val_results_df=test_features(model, datasets, label, scoring, folds)

    data=[list(x) for _, x in cross_val_results_df.groupby('Feature set')['Score']]
    labels=[[x]*len(y) for x, y in cross_val_results_df.groupby('Feature set')['Score']]
    anova_result=f_oneway(*data)
    print(f'ANOVA p-value: {anova_result.pvalue:.3f}\n')

    tukey_result=pairwise_tukeyhsd(np.concatenate(data), np.concatenate(labels), alpha=0.05)

    return cross_val_results_df, tukey_result

# '''Main Execution'''
# if __name__ == "__main__":
#     file_path = "workout_fitness_tracker_data.csv"
#     df = load_data(file_path)
    
#     summarize_data(df)

#     numerical_features = ['Age', 'Height (cm)', 'Weight (kg)', 'Workout Duration (mins)', 'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken']
#     plot_distributions(df, numerical_features)

#     correlation_analysis(df)

#     df = cluster_users(df)
#     df = calculate_efficiency(df)

#     print("Final dataset with efficiency scores:\n", df.head())