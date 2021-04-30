import pandas as pd

import matplotlib.pyplot as plt

insu = pd.read_csv("D:/BLR10AM/Assi/06Hierarchical clustering/Dataset_Assignment Clustering/AutoInsurance.csv")


#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":insu.columns,
                "data types ":insu.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni =insu.nunique()
col_uni


#details of dataframe
insu.describe()
insu.info()

#checking for null or na vales 
insu.isna().sum()
insu.isnull().sum()


#####"Customer ID" is irrelevant and "Count",'Quarter' columns has no variance 
insu_1 = insu.drop(["Customer"], axis=1) # Id# is nothing just index  



########exploratory data analysis

EDA = {"columns_name ":insu_1.columns,
                  "mean":insu_1.mean(),
                  "median":insu_1.median(),
                  "mode":insu_1.mode(),
                  "standard_deviation":insu_1.std(),
                  "variance":insu_1.var(),
                  "skewness":insu_1.skew(),
                  "kurtosis":insu_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(insu_1.iloc[:, :])


#boxplot for every columns
insu_1.nunique()
boxplot = insu_1.boxplot(column=[ "Customer Lifetime Value","Customer Lifetime Value",
                                 "Monthly Premium Auto","Months Since Policy Inception","Total Claim Amount"])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


#unique value for each columns 
insu_1.nunique()

insu_1 = insu_1.iloc[:, [1,8,11,12,13,20,0,2,3,4,5,6,7,9,10,14,15,16,17,18,19,21,22]]
insu_1.info()
# Normalized data frame (considering the numerical part of data)
insu_c_norm = norm_func(insu_1.iloc[:,0:6 ])
insu_c_norm.describe()

#########one hot encoding for discret data
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(insu_1.iloc[:,7:]).toarray())


insu_norm = pd.concat([insu_c_norm,enc_df], axis=1)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(insu_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 1 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering cheacking 8/9/10 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering




########## with 8 cluster 
with_8clust = AgglomerativeClustering(n_clusters = 8, linkage = 'complete', affinity = "euclidean").fit(insu_norm) 
with_8clust.labels_

cluster8_labels = pd.Series(with_8clust.labels_)

insu_1['clust8'] = cluster8_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust8_details  = insu_1.iloc[:, 0:23].groupby(insu_1.clust8).mean()
clust8_details  


########## with 9 cluster 
with_9clust = AgglomerativeClustering(n_clusters = 9, linkage = 'complete', affinity = "euclidean").fit(insu_norm) 
with_9clust.labels_

cluster9_labels = pd.Series(with_9clust.labels_)

insu_1['clust9'] = cluster9_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust9_details  = insu_1.iloc[:, 0:23].groupby(insu_1.clust9).mean()
clust9_details

########## with 7 cluster 
with_10clust = AgglomerativeClustering(n_clusters = 10, linkage = 'complete', affinity = "euclidean").fit(insu_norm) 
with_10clust.labels_

cluster10_labels = pd.Series(with_10clust.labels_)

insu_1['clust10'] = cluster10_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust10_details  = insu_1.iloc[:, 0:23].groupby(insu_1.clust10).mean()
clust10_details


###########final data with 9 clusters 

insu_final = insu_1.iloc[:, [24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
insu_final.head()

# Aggregate mean of each cluster
fclust_details  = insu_final.iloc[:, 1:].groupby(insu_final.clust9).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
insu_final.to_csv("insu_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")