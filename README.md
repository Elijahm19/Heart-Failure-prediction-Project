# Heart_Failure_prediction_Project

Georgia State University 
CSC4780/6780&DSCI4780 – Fundamentals of Data Science  
2022 
Final Project Report 
Heart Failure Prediction
H.E.A.R 
Hanin El-Refai 
Elijah Morton 
Aya Abdelkarem 
Ruqayyah Muse 

Table of Contents 

1 Business Understanding  

1.1 Business Problem 

Heart failure (HF) is a primary cause of death and disability worldwide [1]. An estimated 17.9 million people died from cardiovascular diseases (CVDs) in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke. When it comes to CVDs, heart attacks and strokes are usually acute events caused by a blockage that prevents blood from flowing to the heart or brain. The most common reason for this is a build-up of fatty deposits on the inner walls of the blood vessels that supply the heart or brain. Some risk factors for CVDs are an unhealthy diet, physical inactivity, tobacco use, and the harmful use of alcohol. The effects of behavioral risk factors may be evident in patients through raised blood pressure, increased blood glucose, and overweightness/obesity [1]. 

Considering the effects of CVDs, it is important to detect cardiovascular diseases as early as possible so that management with counseling and medicines can begin. We are confident that building a model on a heart failure prediction dataset will be instrumental in predicting whether someone is likely to have heart disease. With this model, healthcare providers and payers will be able to stratify patients based on the risk of future outcomes and optimize treatment strategies across patients with different needs [2]. Our risk prediction models will also be useful for informing healthcare systems of potential at-risk patients and will enable them to follow up with patients to improve outcomes [2]. 

1.2 Dataset 

The dataset that we are using was found on Kaggle and was originally created using a combination of preexisting data from different cities and countries. The dataset was filtered to remove duplicate entries resulting in the final data having 918 instances, 11 descriptive features, and 1 target feature [3]. The target feature is titled HeartDisease and is a Boolean with a domain of 1 and 0, 1 being the patient has heart disease, and 0 being the patient does not have heart disease. 

As for the descriptive features: 

The age feature gives the ages of the patients.  

The sex feature represents the gender of the patients. 

The ChestPainType feature gives the type of chest pain, which can be TA meaning Typical Angina, ATA meaning Atypical Angina, NAP meaning Non-Anginal Pain, or ASY meaning Asymptomatic. Angina is defined as a “type of chest pain caused by reduced blood flow to the heart” [4]. Typical angina is usually preceded by physical exertion or stress and is relieved with rest or medication. Atypical angina is described as having symptoms that are ascribed to angina but don’t fall under the category of typical angina [5]. Non-Anginal Pain refers to pain in the chest area that is not related to the heart [6]. Asymptomatic is when there are no symptoms being shown. 

Resting BP stands for resting blood pressure which is the blood pressure when your heart is at rest. It is measured by millimeters (mm) of mercury (Hg).   

Cholesterol gives the serum cholesterol level which is the amount of total cholesterol in your blood [7]. It is measured by milligrams (mg) of cholesterol per deciliter (dL) of blood (mm/dl).  

Fasting BS is fasting blood sugar which is your blood sugar level after an overnight period of no eating [8].  

Resting ECG is resting electrocardiogram results which can be Normal, LVH, or ST. Resting ECG is a test that can help detect heart abnormalities like arrhythmia and show evidence of heart disease and left ventricular hypertrophy (LVH) [9]. ST refers to ST-T wave abnormality which is when there is an elevation or depression of the region between the ventricular end of depolarization and the beginning of repolarization and/or T wave inversions [10]. This can be indicative of myocardial ischemia or myocardial injury which are conditions related to the heart [10]. Resting ECG measured in millivolts (mV). 

MaxHR is the maximum heart rate achieved.  

ExerciseAngina refers to whether angina (see feature 3 above) was induced by exercise or not.  

Oldpeak is the measure of ST depression induced by exercise relative to rest. It is measured based on levels of depression. 

ST_Slope is the ST segment's slope which measures the segment's shift “relative to exercise-induced increments in heart rate” [11]. It can be up-sloping, flat, or down-sloping. 

With these features and 918 already classified entries, we can use this dataset for predicting heart disease in a patient. 

1.3 Proposed Analytics Solution 

In the dataset, there are 11 attributes and the target class which is ‘heart disease’ is a binary class. The value 1 represents the presence of heart disease while 0 means that the person doesn’t have the disease. In this case, the proposed analytics solution is applying classification machine learning models such as Classifier Decision Trees, Logistic Regression, mutual information, and K-Nearest Neighbor (KNN).  
 
 

2 Data Exploration and Preprocessing 

2.1 Data Quality Report 

The data quality report shows that there are no missing values in any of the data features.  

 

Table 1. Data Quality Report for Categorical Features 

 

There are seven categorical data features; six are descriptive features and one is the target variable.  

The Sex feature takes F or M values, F for female and M for male, making the cardinality of this feature two. The mode of the Sex feature is M with a frequency of 725 which means that males occupy around 79% of the dataset.  

ChestPainType has a cardinality of four since its domain is TA, ATA, NAP, and ASY. ASY (Asymptomatic) chest pain type is the most frequent value. 54% of the patients have asymptomatic chest pain. 

FastingBS is a Boolean with a domain of 0 or 1, so its cardinality is two. It has a value of 1 if the fasting blood sugar is greater than 120 mg/dl and 0 otherwise. Zero is the mode of this feature with 77% of the instances having fasting blood sugar levels that are less than or equal to 120 mg/dl.  

RestingECG has a domain of Normal, ST, and LVH making the cardinality three. The mode of this feature is Normal, as patients with Normal ECG results make up 60% of the instances. 

ExerciseAngina has a cardinality of two since its domain is Yes or No. The mode of this feature is No. About 60% of the Angina cases were not induced by exercise.  

The ST-Slope feature has a cardinality of three: up, flat, and down. Half of the dataset had a flat slope during peak exercise.  

Finally, HeartDisease, which is the target feature, is a Boolean with a cardinality of two. 44.6% of the patients have heart disease. 

The remaining descriptive features are continuous ones.  

Age shows the age of the patient, and the values range from 28 to 77, with a mean of 53.5. 

The RestingBP attribute has a domain of 0 to 200. The values signify the presence of outliers, since a zero-blood pressure means a dead person and 200 is an extreme blood pressure value. 

The Cholesterol attribute has a domain of 0 to 603. The zero values and the high range indicate the presence of outliers, as no human can have a zero-cholesterol level. This value could have been inputted as zero to fill in untested cholesterol levels for some patients. Additionally, the standard deviation for cholesterol levels is 109.4 which is very high. 

MaxHR ranges between 60 and 202 with a mean of 136.8 and a standard deviation of 25.5. 

Oldpeak has a minimum of -2.6 and a maximum of 6.2 with a 1.07 standard deviation.   

Analyzing Bar Charts for Categorical Features 

The bar plot for the Sex feature shows that the number of males is almost four times the number of females in the dataset. The ASY (Asymptomatic) chest pain type is the most prevalent type of chest pain at almost 400 instances, while TA (Typical Angina) scores the lowest at a level lower than 100. The number of instances where fasting blood pressure is less than or equal to 120 mg/dl is 3.5 times more than the number of instances where fasting blood pressure is greater than 120 mg/dl.  


 
Additionally, the number of instances with Normal resting ECG is larger than the number of instances of LVH (which shows probable or definite left ventricular hypertrophy by Estes’ criteria) and ST (which shows the patient is having wave abnormality with elevation or depression of more than 0.05 mV). This means that there might be people with heart disease that have normal resting ECG. As for ExerciseAngina, the number of cases with “No” Exercise Angina is around 375, which is higher than the number of cases of Exercise Angina. Instances with flat ST_Slope (zero slope) are the largest number of instances compared to upward or downward slopes. Finally, the number of instances of heart disease is slightly higher than instances of no heart disease with a difference of 100 instances.  

 
The histogram of the Age feature shows a unimodal normal distribution. On the other hand, RestingBP follows a unimodal right-skewed distribution, which will require normalization before applying any machine learning model. Also, 350 instances fall around 125 mm/Hg RestingBP. As for the Cholesterol feature, there are 175 instances with 0 to 50 mm/dl cholesterol levels which shows the presence of outliers in the feature. The rest of the data has a right-skewed unimodal distribution with up to 600 mm/dl cholesterol levels, reflecting large outlier values. For the MaxHR feature, it follows a multimodal distribution with two peaks at 120 and 140, with 120 instances at each peak. The Oldpeak feature follows a unimodal right-skewed distribution with the highest concentration being at 0.  

Analyzing Box Plots for Continuous Features 


The box plots for the continuous data features show that there are outliers for all continuous features except Age. It is an indication that we need to handle the outliers before applying any machine learning models. RestingBP has an outlier at zero and upper outliers with values greater than 175 mm/Hg. The rest of the values are right skewed towards 125 mm/Hg. The boxplot of the Cholesterol feature shows outliers at both limits, zero at the lower limit, and values above 400 mm/dl at the upper limit. MaxHR has outliers at the bottom limit with two instances having values less than 67. Finally, the boxplot of Oldpeak has outliers at both limits, one being a value less than –2.5 and seven being greater than 3.8.  

2.2 Missing Values and Outliers 

The data does not have any missing values. However, the boxplots and the histograms revealed the presence of outliers in all the continuous descriptive features except for Age. Therefore, the outliers are handled by clamping, using the IQR method. To clamp outliers, the first and third quartiles are calculated, Q1 and Q3 respectively. The IQR is then obtained by subtracting Q1 from Q3 (Q3 – Q1). The lower limit is set to Q1 – 1.5IQR, while the upper limit is set to Q3 + 1.5IQR. Clamping is the process of setting the outliers that are less than the lower limit equal to the value of the lower limit, while the ones greater than the upper limit are set equal to the value of the upper limit. Using this method does not remove any data, so there is no loss of information. 

2.3 Normalization 

We used range normalization on all the continuous descriptive features so that their values would fall between 0 and 1. Originally the domain of each feature was vastly different, with Cholesterol having an upper limit of around 600 while values for Oldpeak were 6 or less. Features with much larger numbers might overly influence the classification even though they might not be that important which is why we normalized the ranges.  

2.4 Transformations 

Most machine learning models require transforming categorical features and encoding them into numbers. Therefore, one-hot encoding is applied using the get_dummies function. This was done to ensure maximum data quality, which is imperative for gaining accurate analysis. 


2.5 Feature Selection 

Feature selection helps to simplify the models to make them easier to interpret, have shorter training times, and to improve data compatibility with a learning model class. We applied two feature selection methods that we found compelling with our data set. The first algorithm we used was Recursive Feature Elimination or RFE for short and the second algorithm was Mutual information classifier. The RFE is popular because it is easy to configure and is effective at selecting features in the training set that are more effective at selecting those features that can predict the target variable. We did configuration to help us choose the number of features to like. We selected 8 features that had the first rank to predict the target variable: Age, RestingBP, Cholesterol, MaxHR, OldPeak, ChestPainType_TA,  ChestPainType_ASY, and ST_Slope_Up after hot encoding the data. The second feature selection Mutual information classifier is the amount of information one variable gives about the target variable. Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency. It can be used for univariate feature selection. Based on our model after applying the mutual information method we select the top 8 features with the highest or strongest dependencies and these were: ST_Slope_Up, ST_Sope_Flat, Oldpeak, ChestPainType_ASY, ChestPainType_ATA, ExerciseAngina_N, ST_Slope_Down, ExerciseAngina_Y. Both feature selection methods resulted into choosing different features but also there were common features that the two methods chose. Therefore, we chose to use ST_Slope_Up, ChestPainType_ASY, Oldpeak, MaxHR, Sex_M, Cholesterol, and age as they are some of the better-ranked features given the two feature selection methods we applied. 

3 Model Selection and Evaluation 

Since the problem is a classification problem, four classification models were applied to classify patients. All models applied are imported from scikit-learn library. 

3.1 Evaluation Metrics 

For evaluation metrics, we are using the accuracy score and confusion matrix calculations from the SciKit Learn Library. The accuracy score tells us how well each model accurately predicts the classification for each patient. The confusion matrix is also handy for seeing how many false positives and false negatives the model predicted, which is important for our business problem because we are dealing with patient diagnoses. We need to understand the type of misclassifications we have to minimize the number of false negatives which would suggest that a patient doesn’t have heart disease when in fact, they do. 

We also are using FAR which is the False Alarm Ratio, CSI which is the threat score, TSS which is the true skill statistic, and the bias score as additional evaluation metrics. The false alarm ratio indicates the percentage of false positives where a patient does not have heart disease but is classified as having it. The threat score indicates the percentage of actual heart disease cases that were correctly predicted. The true skill statistic measures how well the model differentiates between correctly predicted cases of heart disease and cases where heart disease does not occur. Finally, the bias score measures the ratio of predicted cases of heart disease to true cases of heart disease. 

Based on these definitions, the best models would be the ones that minimize the false alarm ratio while maximizing the threat score, true skill statistic, and bias score. 

3.2 Models 

 	The first model applied is Decision Tree Classifier (DT). It is set with maximum depth of five and the splitting criterion was entropy. In this case, DT calculates the entropy of “Heart Disease” feature. Then it calculates the entropy of each of the descriptive features taking the weights of each level of the descriptive features into account. Information gain is then calculated by subtracting results (remainder) of descriptive features from entropy of target feature. The feature with the highest information gain is chosen as the root node. Splitting keeps occurring until the maximum depth defined, five, is reached. A DT instance is created and fitted to the training data. The input to the DT is training data, and the output is a label or class based on a probability vector produced by the DT. This vector includes a probability value for each level of the target feature, and the level with maximum probability is chosen to be the predicted class.  

The second model applied is K-Nearest Neighbor Classifier. It is first applied with one nearest neighbor. Since k is the number of neighbors, a visualization of k, accuracy score of KNN on training set, and accuracy score of KNN on testing set is created to obtain the optimal k. The visualization showed that 15 is the optimal number of k; therefore, KNN is applied with k = 15. Also, the type of distance referred to as p in a KNN can be specified. The default model uses p = 1 which is Manhattan distance. The applied KNN model uses Manhattan distance. A KNN instance is initialized and fitted to the training dataset. The output of KNN is a distance between the data point to be predicted and the training data points. The classes of data points with the least minimum distance are chosen based on a probability vector. Like DT, the probability vector is a two-dimensional vector with a probability for each of the classes, and the class with higher probability is chosen as the label.  

The third possible model is Gaussian Naïve Bayes which applies Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features [12]. Naïve bayes produces probability vectors for tested data point based on levels of target feature “Heart Disease”. The level with highest probability value is voted to be the class for the test data point.  

The last applied model is Logistic regression. Logistic regression is an error-based learning model that finds the optimal line that can separate the data into one positive class and another negative class. This line is obtained by calculating optimal weights that can result in the least possible error. Gradient Descent is the method used in the case to search the convex error surface for the global minimum based on a learning rate C. A Logistic Regression instance is created and fitted to the data. The inputs are training data points, and the outputs are probability estimates to classify based on higher probability.  

Predictions from each model are issued and model performance is evaluated. 

3.3 Evaluation 

3.3.1 Evaluation Settings and Sampling 

We imported the ‘train_test_split’ method from the Sci-Kit Learn library. We then split the data into a training set and a testing set. The training set size represents two thirds of the data (67%), while the rest of the data is the test set size (33%).  

3.3.2 Hyper-parameter Optimization 

We used Grid Search for hyper parameter optimization. Before tuning our KNN Classifier, our accuracy was 0.77 respectively, and after tuning our parameters, our accuracy increased to 0.85. Additionally, Prior to tuning our Decision Tree classifier, our accuracy was 0.82 respectively, after tuning our parameters, our accuracy increased to 0.84. Finally, before tuning our Logistic Regression model, our accuracy was 0.83 respectively, after tuning our parameters, our accuracy increased to 0.86. We can see that after tuning our parameters to these three models, their accuracies increased slightly. 

3.3.3 Evaluation 

After tuning the hyperparameters of the models, DT accuracy increased from 81% to 82%. KNN’s accuracy improved to become 84.2% from a previous accuracy of 76.9, while Logistic Regression scored the same as before tuning at 83%. Since the hyperparameters for Naïve Bayes were not tuned, its accuracy stayed the same at 87%. Therefore, the best-performing model is Gaussian Naïve Bayes.  

4 Results and Conclusion 

Based on models’ performances, Gaussian Naïve Bayes is the one chosen to best fit the data and the analytical business problem. Given the importance of accurate predictions for this business problem, heart disease prediction requires the least amount of false negative predictions. Therefore, choosing to deploy Gaussian Naïve Bayes model seems to be the optimal choice. However, in the long run, there should be awareness of possible concept-drift, where the model would not be performing the same after deployment. The same evaluation metrics used before deployment should be used to evaluate the model after deployment, and any significant changes in those metrics, the model should be revisited, updated, and reevaluated. For instance, if accuracy decreases below 84%, that should be considered as a significant change that needs to be addressed.  

5 Reference 

[1] World Health Organization. (n.d.). Cardiovascular diseases (cvds). World Health Organization. Retrieved November 4, 2022, from https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)  

[2] Di Tanna, G. L., Wirtz, H., Burrows, K. L., & Globe, G. (2019). Evaluating risk prediction models for adults with heart failure: A systematic literature review. PLoS ONE, 15(1). https://doi.org/10.1371/journal.pone.0224135 

[3] Fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved Oct 25, 2022, from https://www.kaggle.com/fedesoriano/heart-failure-prediction. 

[4] Mayo Clinic. (2022, March 30). Angina - Symptoms and Causes. Retrieved November 4, 2022 from https://www.mayoclinic.org/diseases-conditions/angina/symptoms-causes/syc-20369373  

[5] AlBadri, A., Leong, D., Bairey Merz, C. N., Wei, J., Handberg, E. M., Shufelt, C. L., Mehta, P. K., Nelson, M. D., Thomson, L. E., Berman, D. S., Shaw, L. J., Cook‐Wiens, G., & Pepine, C. J. (2017). Typical angina is associated with greater coronary endothelial dysfunction but not abnormal vasodilatory reserve. Clinical Cardiology, 40(10), 886–891. Retrieved November 4, 2022 from https://doi.org/10.1002/clc.22740  

[6] Non-Cardiac Chest Pain. (n.d.). Cleveland Clinic. Retrieved November 4, 2022 from https://my.clevelandclinic.org/health/diseases/15851-gerd-non-cardiac-chest-pain  

[7] Heart UK. (2022). Understanding your cholesterol test results. Heart UK The Cholesterol Charity. Retrieved November 4, 2022 from https://www.heartuk.org.uk/cholesterol/understanding-your-cholesterol-test-results-  

[8] CDC. (2019, May 15). Diabetes Testing. Centers for Disease Control and Prevention. Retrieved November 4, 2022 from https://www.cdc.gov/diabetes/basics/getting-tested.html#:~:text=Fasting%20Blood%20Sugar%20Test  

[9] Centre (UK), N. G. (2016). Resting electrocardiography. National Library of Medicine; National Institute for Health and Care Excellence (UK). Retrieved November 4, 2022 from https://www.ncbi.nlm.nih.gov/books/NBK367910/  

[10] Kashou, A. H., & Kashou, H. E. (2019). Rhythm, ST Segment. National Library of Medicine; StatPearls Publishing. https://www.ncbi.nlm.nih.gov/books/NBK459364/  

[11] Finkelhor, R. S., Newhouse, K. E., Vrobel, T. R., Miron, S. D., & Bahler, R. C. (1986). The ST segment/heart rate slope as a predictor of coronary artery disease: comparison with quantitative thallium imaging and conventional ST segment criteria. American Heart Journal, 112(2), 296–304. Retrieved November 7, 2022 from https://doi.org/10.1016/0002-8703(86)90265-6 

[12] Scikit- Learn Documentation. (n.d.). Naïve Bayes. Retrieved December 2, 2022 from https://scikit-learn.org/stable/modules/naive_bayes.html. 
