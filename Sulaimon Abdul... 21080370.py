
# import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import err_ranges as err
from scipy.optimize import curve_fit

# a function that reads a dataframe
def solution(filename, cols):
    '''
    This function returns a dataframe and transpose dataframe and accepts the following arguments
    filename: this is the url of the data file to be used for analysis
    'cols: these are the features to be dropped'
    '''
    df = pd.read_excel(filename, sheet_name = 'Data', skiprows=3)
    df = df.drop(cols, axis=1)
    df.set_index('Country Name', inplace = True)
    return df, df.T

# the attributes are passed into the function
filename = 'https://api.worldbank.org/v2/en/indicator/SP.POP.GROW?downloadformat=excel'
cols = ['Country Code', 'Indicator Name', 'Indicator Code', '1960']

data_pop, data_pop_transpose = solution(filename, cols)
print(data_pop)

# the dataframe for clustering is extracted
df_clustering = data_pop.iloc[:, [0,-1]]
print(df_clustering)

# check for null values
df_clustering.isnull().sum()

# drop null values
data_cluster = df_clustering.dropna()
data_cluster

# create a scatter plot of the features
plt.figure(figsize=(12,8))
plt.scatter(data_cluster['1961'], data_cluster['2021'])
plt.title('Scatter plot of population growth involving 1961 and 2021', fontsize=20)
plt.xlabel('1961', fontsize=15)
plt.ylabel('2021', fontsize=15)
plt.show()

# from the look of the dataframe, the data appear scaled
# create a plot that depicts the ideal number of clusters

sse = [] # sum of squared errors
for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_cluster)
        sse.append(kmeans.inertia_)

plt.figure(figsize=(10,8), dpi=200)
plt.plot(range(1, 11), sse) # visualize the elbow
plt.title('Elbow of clusters', fontsize=20)
plt.xlabel('number of clusters', fontsize=15)
plt.ylabel('sse', fontsize=15)
plt.show()

# from the elbow graph, the dataframe should be grouped into 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(data_cluster)
y_pred = kmeans.fit_predict(data_cluster)

# cluster centers
cluster_centers = kmeans.cluster_centers_
cluster_centers

# silhouette score
silh_score = silhouette_score(data_cluster, y_pred)
silh_score

# create a scatter plot that visualizes the clusters and the centroids of each cluster

plt.figure(figsize=(12,8), dpi=200)
plt.scatter(data_cluster[y_pred == 0].iloc[:,0], data_cluster[y_pred == 0].iloc[:,-1], s = 50, c = 'red', label='K1')
plt.scatter(data_cluster[y_pred == 1].iloc[:,0], data_cluster[y_pred == 1].iloc[:,-1], s = 50, c = 'blue', label='K2')
plt.scatter(data_cluster[y_pred == 2].iloc[:,0], data_cluster[y_pred == 2].iloc[:,-1], s = 50, c = 'yellow', label='K3')
plt.scatter(data_cluster[y_pred == 3].iloc[:,0], data_cluster[y_pred == 3].iloc[:,-1], s = 50, c = 'green', label='K4')
plt.scatter(data_cluster[y_pred == 4].iloc[:,0], data_cluster[y_pred == 4].iloc[:,-1], s = 50, c = 'purple', label='K5')
plt.scatter(cluster_centers[:, 0], cluster_centers[:,1], s = 100, c = 'black', label = 'Centroids')
plt.title('Scatterplot showing clusters and centroids of Population growth between 1961 and 2021', fontsize=15, fontweight='bold')
plt.xlabel('1961', fontsize=15)
plt.ylabel('2021', fontsize=15)
plt.legend(bbox_to_anchor=(1.14,1.01))
plt.show()

# create a column for the clusters
data_cluster['label'] = y_pred

# the first, second and fourth cluster is used for visualization 
first_cluster = data_cluster[y_pred == 0].iloc[-5:]
second_cluster = data_cluster[y_pred == 1].iloc[:,:2]
fourth_cluster = data_cluster[y_pred == 3].iloc[:5]

# the first and fourth cluser data are printed
print(first_cluster)
print(fourth_cluster)

# a grouped bar plot is created for the first cluster
arrays = ['Uganda', 'Vanuatu', 'Yemen', 'Zambia', 'Zimbabwe']
x = np.arange(len(arrays)) # x is the range of values using the length of the arrays
width = 0.2
fig, ax  = plt.subplots(figsize=(12,8))
    
plt.bar(x - width, first_cluster['1961'], width, label='1961') 
plt.bar(x, first_cluster['2021'], width, label='2021')   
plt.title('Population growth of 5 countries using first cluster between 1961 and 2021', fontsize=15)
plt.ylabel('Population growth', fontsize=15)
plt.xticks(x, arrays)
plt.legend()
ax.tick_params(bottom=False, left=True)

plt.show()

# create a grouped bar plot for the fourth cluster
arrays = ['Argentina', 'Antigua and Barbuda', 'Australia', 'Austria', 'Belgium']
x = np.arange(len(arrays)) # x is the range of values using the length of the arrays
width = 0.2
fig, ax  = plt.subplots(figsize=(12,8))
    
plt.bar(x - width, fourth_cluster['1961'], width, label='1961') 
plt.bar(x, fourth_cluster['2021'], width, label='2021')   
plt.title('Population growth of 5 countries using fourth cluster countries between 1961 and 2021', fontsize=15)
plt.ylabel('Population growth', fontsize=15)
plt.xticks(x, arrays)
plt.legend()
ax.tick_params(bottom=False, left=True)

plt.show()

# print the second cluster data
print(second_cluster)

# create a heatmap for the year 1961 and 2021
plt.figure(figsize=(12,8))
sns.heatmap(second_cluster)
plt.title('Heatmap between 1961 and 2021 for second cluster')
plt.show()

# a dataframe is created using Canada as its reference
df_Canada = pd.DataFrame({
    'Year' : data_pop_transpose.index,
    'Canada' : data_pop_transpose['Canada']
})
df_Canada.reset_index(drop=True)

# change the data type to int64
df_Canada['Year'] = np.asarray(df_Canada['Year'].astype(np.int64))

# create a plot for the Canada dataframe
plt.figure(figsize=(12,8), dpi=100)
plt.plot(df_Canada['Year'], df_Canada['Canada'])
plt.title('Time series plot showing population growth in Canada', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Growth', fontsize=15)
plt.show()

# a function that returns an exponential equation
def model_fit(a, b, c):
    ''' this function calculates the exponential growth with a scale factor b and c'''
    a = 1961 - a
    y = b * np.exp(c*a)
    return y

# the parameters and the covariance of the function are passed into the curve fit method
param, cova = curve_fit(model_fit, df_Canada['Year'], df_Canada['Canada'])
print(param)

print(cova)

# a plot of the forecast, confidence intervals is displayed below
year = np.arange(1961, 2032) 
sigma = np.sqrt(np.diag(cova))
forecast = model_fit(year, *param)
lower, upper = err.err_ranges(year, model_fit, param, sigma)

plt.plot(df_Canada["Year"], df_Canada["Canada"], label="Canada")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, lower, upper, color="yellow", alpha=0.7)
plt.title('A plot showing the confidence ranges, forecast of population growth in Canada')
plt.xlabel("year")
plt.ylabel("Canada")
plt.legend()
plt.show()

# create a dataframe that predicts values for the next 10 years

predictions = pd.DataFrame({
                            'year': year,
                            'forecast': forecast
})
prediction_ten_years = predictions.iloc[61:]
print(prediction_ten_years)



