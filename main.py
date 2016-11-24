# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
# %matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data2 = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# Display a description of the dataset
display(data.describe())

#  Select three indices of your choice you wish to sample from the dataset
indices = [10, 100, 356]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data2.loc[indices], columns = data2.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

############################ FEATURE RELEVANCE #####################################################
#  Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop("Grocery", axis=1)

#  Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
np.random.seed(50)  # Initialize the random number generator
X_train, X_test, y_train, y_test = train_test_split(new_data, data["Grocery"], test_size=0.25)

#  Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#  Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print "R^2 score predicting Detergents_Paper : %f" % score

# Produce a scatter matrix for each pair of features in the data
# pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

#  Scale the data using the natural logarithm
log_data = np.log(data)

#  Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
# pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


############################ REMOVING OUTLIERS #####################################################

outliers_indices = []  # This list will be filled with all the indices of the outliers

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():

    #  Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)

    #  Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)

    #  Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)

    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

    # Adding outliers indices to the list
    outliers_indices += log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.tolist()

# OPTIONAL: Select the indices for data points you wish to remove
outliers = [x for x in outliers_indices if outliers_indices.count(x) < 2]  # Remove the outlier that have been detected in more than two features.

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


#if not log_samples.isin(good_data):
#    print log_samples.isin(good_data)
#    raise SystemExit(0)


############################ PCA #####################################################

from sklearn.decomposition import PCA
#  Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)
pca = pca.fit(good_data)

#  Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = rs.pca_results(good_data, pca)

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

#  Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca = pca.fit(good_data)

#  Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

#  Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

############################ CLUSTERING #####################################################


def kmeans_clustering():
    '''KMeans clustering.'''
    #  Apply your clustering algorithm of choice to the reduced data
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=2)
    clusterer.fit(reduced_data)

    #  Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    #  Find the cluster centers
    centers = clusterer.cluster_centers_

    #  Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    #  Calculate the mean silhouette coefficient for the number of clusters chosen
    from sklearn.metrics import silhouette_score
    score = silhouette_score(reduced_data, preds)
    print "Score : %f" % score
    return preds, centers, score


def gmm_clustering():
    '''GMM.'''
    from sklearn.mixture import GMM
    g = GMM(n_components=4)
    g.fit(reduced_data)
    preds = g.predict(reduced_data)
    sample_preds = g.predict(pca_samples)
    from sklearn.metrics import silhouette_score
    score = silhouette_score(reduced_data, preds)
    print "Score : %f" % score


preds, centers, score = kmeans_clustering()

#  Inverse transform the centers
log_centers = pca.inverse_transform(centers)

#  Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
