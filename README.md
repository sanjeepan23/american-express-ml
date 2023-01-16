# American-Express-ML

##vMethodology
Since the dataset is huge all float68 columns were converted into float32 columns which reduces the size and the modified dataset was able to fit into the memory. It’s observed that there are multiple columns that contain large amounts of NaN values, initially, all the columns that have NaN values of more than 40% were dropped then the frequency Imputation method was used to fill the nan values of other columns. Then IRQ method was used to remove all the outliers outside the 75th and 25th percentile. 

After that one-hot encoding was performed on categorical columns then all the numerical columns were standardized using StandardScaler. Then the correlation between the column was calculated and all the columns that were correlated above 85% dropped. Similarly, the variance was calculated and all the columns with less variance (0.05) dropped which leads to a column count of 196.

Since the dataset is imbalanced, StratifiedKFold was used to create cross-validation folds and three models were trained namely XGBoost, SVM, and KNN. However, SVM and KNN didn’t perform well on the baseline experiment but XGBoost performed exceptionally well and yielded an average cross-validation score of 0.8552. Finally, a randomized search was used to find optimal parameters for the data.

