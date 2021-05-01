
import pandas as pd

import matplotlib.pylab as plt

air = pd.read_excel(r"D:\\BLR10AM\\Assi\\06Hierarchical clustering\\Dataset_Assignment Clustering\\EastWestAirlines.xlsx",sheet_name="data")


#######feature of the dataset to create a data dictionary
description  = ["Unique ID",
                "Number of miles eligible for award travel",
                "Number of miles counted as qualifying for Topflight status",
                 "Number of miles earned with freq. flyer credit card in the past 12 months:",
                 "Number of miles earned with Rewards credit card in the past 12 months:",
                 "Number of miles earned with Small Business credit card in the past 12 months: 1 = under 5,000 2 = 5,000 - 10,000 3 = 10,001 - 25,000  4 = 25,001 - 50,000 5 = over 50,000",
                  "Number of miles earned from non-flight bonus transactions in the past 12 months",
                 "Number of non-flight bonus transactions in the past 12 months",
                "Number of flight miles in the past 12 months",
                "Number of flight transactions in the past 12 months",
                "Number of days since Enroll_date",
                "Dummy variable for Last_award (1=not null, 0=null)"]

d_types =["count","ratio","continuous","count","count","count","continuous","continuous","continuous","continuous","continuous","count"]

data_details =pd.DataFrame({"column name":air.columns,
                "description":description,
                "data types ":d_types})


###########Data Pre-processing 

#unique value for each columns 
col_uni =air.nunique()

#details of dataframe
air.describe()
air.info()

#checking for null or na vales 
air.isna().sum()
air.isnull().sum()



air_1 = air.drop(["ID#"], axis=1) # Id# is nothing just index  



########exploratory data analysis

EDA = {"columns_name ":air_1.columns,
                  "mean":air_1.mean(),
                  "median":air_1.median(),
                  "mode":air_1.mode(),
                  "standard_deviation":air_1.std(),
                  "variance":air_1.var(),
                  "skewness":air_1.skew(),
                  "kurtosis":air_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(air_1.iloc[:, :])


#boxplot for every column
for column in air_1:
    plt.figure()
    air_1.boxplot([column])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


#unique value for each columns 
air.nunique()

air_1 = air_1.iloc[:, [0,1,5,6,7,8,9,10,2,3,4]]
air_1.nunique()
# Normalized data frame (considering the numerical part of data)
air_c_norm = norm_func(air_1.iloc[:,0:7 ])
air_c_norm.describe()

#########one hot encoding for discret data
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(air_1.iloc[:,7:]).toarray())


air_norm = pd.concat([air_c_norm,enc_df], axis=1)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(air_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 1 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering cheacking 4/5/6 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering



########## with 4 cluster 
with_4clust = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(air_norm) 
with_4clust.labels_

cluster4_labels = pd.Series(with_4clust.labels_)

air_1['clust4'] = cluster4_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust4_details  = air_1.iloc[:, 0:12].groupby(air_1.clust4).mean()
clust4_details

########## with 5 cluster 
with_5clust = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(air_norm) 
with_5clust.labels_

cluster5_labels = pd.Series(with_5clust.labels_)

air_1['clust5'] = cluster5_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust5_details  = air_1.iloc[:, 0:12].groupby(air_1.clust5).mean()
clust5_details  


########## with 6 cluster 
with_6clust = AgglomerativeClustering(n_clusters = 6, linkage = 'complete', affinity = "euclidean").fit(air_norm) 
with_6clust.labels_

cluster6_labels = pd.Series(with_6clust.labels_)

air_1['clust6'] = cluster6_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust6_details  = air_1.iloc[:, 0:12].groupby(air_1.clust6).mean()
clust6_details



###########final data with 5 clusters 

air_final = air_1.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10]]
air_final.head()

# Aggregate mean of each cluster
fclust_details  = air_final.iloc[:, 1:].groupby(air_final.clust5).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
air_final.to_csv("air_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")

