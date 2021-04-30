# Load the dataset
library(readr)
insu <- read_csv(file.choose())
insu_1 <- insu[ , c(2:24)]

summary(insu_1)


library(fastDummies)

#columns name 
names(insu_1)

###### Normalization ###################################
# categorical data 
data_dummy  <- dummy_cols(insu_1, select_columns = c("State","Response"    ,   "Coverage" ,"Education"    ,  "Effective To Date","EmploymentStatus","Gender","Location Code" ,"Marital Status","Number of Open Complaints",  "Number of Policies","Policy Type","Policy", "Renew Offer Type" ,"Sales Channel"  ,"Vehicle Class","Vehicle Size"),
                         remove_first_dummy = TRUE,remove_most_frequent_dummy = FALSE,remove_selected_columns = TRUE)

# Normalize the data
# to normalize the data we use custom function 
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

df_norm <- as.data.frame(lapply(data_dummy, norm)) # Excluding the nominal column




summary(df_norm)

# Distance matrix dis= euclidean 
d <- dist(df_norm, method = "euclidean") 

#linkage = complete
fit <- hclust(d, method = "complete")

# Display dendrogram
 
plot(fit, hang = -77)


#######after looking at dendrogram we go eith 9 clusters 
groups <- cutree(fit, k = 9) # Cut tree into 9 clusters

rect.hclust(fit, k = 9, border = "red")

membership <- as.matrix(groups)

#creating a final data frame with cluster num 
final <- data.frame(membership, insu_1)

#details of each group
fclust_details <-aggregate(insu_1[,c(2,9,12,13,14,15,16,21)], by = list(final$membership), FUN = mean)
fclust_details
#saving the result
write_csv(final, "hclustoutput.csv")

