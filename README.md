# Home Loan Qualification Prediction Using Supervised Machine Learning Algorithms

 The machine learning problem was solved by following the Cross Industry Process Standards for Data Mining (CRISP-DM) Framework as outlined below:

<img width="586" alt="Screenshot-2016-04-20-11 58 54" src="https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/143b3816-e862-4039-aab2-7f10e0c02c65">

# Step 1: Business Understanding

### Background

Dream Housing Finance Company is a leading provider of home loans with a wide presence across urban, semi-urban, and rural areas. Customers first apply for home loans, and the company then validates their eligibility. The company aims to improve its loan eligibility process by leveraging machine learning algorithms.

### Determine Business Objectives

The primary objective is to automate the loan eligibility process in real-time based on customer details provided during the online application. The business success criteria include:

* Efficiency Improvement: Automate the eligibility process to reduce processing time by at least 50%.
* Scalability: Enable the company to handle a larger volume of loan applications efficiently.
* Accuracy: Achieve an accuracy rate of at least 85% in predicting loan eligibility.
* Cost Reduction: Reduce operational costs associated with manual processing.
* Customer Satisfaction: Improve customer satisfaction scores through faster and more reliable eligibility decisions.

### Assess Situation

* Resources Availability: The company has access to historical loan application data, including customer details and loan approval status.
* Project Requirements: The project requires data preprocessing, feature selection, model training, and validation. It also involves integration with the existing application system for real-time processing.
* Risks and Contingencies: Potential risks include data quality issues, model overfitting or underfitting, and integration challenges with the current system. Contingency plans involve rigorous data cleaning, regular model evaluation, and phased integration.
* Cost-Benefit Analysis: The benefits of automating the loan eligibility process include significant time savings, reduced operational costs, improved accuracy, and enhanced customer satisfaction. These benefits outweigh the initial costs of developing and implementing the machine learning model.

### Determine Data Mining Goals

From a technical perspective, the data mining goals include:

* Data Collection and Preparation: Gather and preprocess historical loan application data.
* Feature Engineering: Identify and create relevant features such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, and Credit History.
* Model Training and Evaluation: Train multiple supervised classification algorithms and evaluate their performance to select the best model.
* Real-Time Prediction: Develop a system for real-time loan eligibility prediction based on the selected model.

### Produce Project Plan

Technology and Tools Selection: Use R for data processing and machine learning, and Shiny for integrating the model with the online application system.
Detailed Project Phases:
* Data Collection and Preprocessing: Collect data, handle missing values, and preprocess for modeling.
* Feature Engineering: Create and select relevant features.
* Model Training and Evaluation: Train and evaluate various machine learning models using R.
* Model Integration: Integrate the chosen model with the existing system for real-time prediction using Shiny.
* Deployment and Monitoring: Deploy the system and monitor its performance regularly.

# Step 2: Data Understanding

### Data Collection

We are working with historical loan data from Dream Housing Finance Company. This dataset includes various attributes related to customers and their loan applications. Our goal is to characterize and understand these variables to inform the subsequent steps in the data mining process.

### Base Table Creation

The first step in understanding our data was to create a base table that consolidates all relevant variables. This base table includes attributes such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others.

Observing the first 6 rows of the data gives a high level view of the variables 

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/f019d59c-d744-41db-be90-c653b5230e6d)

The dataset has 614 Observations and 13 variables. The Loan_ID id variable is dropped as it adds no significant meaning to the machine learning performance.The target variable is the Loan_Status that states if a client was provided with a home loan or not.

#### Checking the number of missing values per variable

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/ee25ad0e-1bca-4ed6-ac48-87e889cdfc95)

The analysis of the provided dataset, which contains a total of 614 observations, reveals several instances of missing values across various key variables. Addressing these missing values is critical for ensuring the robustness and reliability of the machine learning models developed from this data.

The variable "Self_Employed" exhibits 32 missing values, which constitutes approximately 5.2% of the dataset. The absence of employment status information could potentially obscure important correlations between self-employment and loan eligibility. To mitigate this, imputation strategies such as replacing missing values with the most frequent category (mode) or introducing a new category to account for unknown employment statuses should be considered.

The "LoanAmount" variable shows 22 missing values, representing about 3.6% of the dataset. Given that the loan amount is a critical factor in determining loan eligibility, the absence of this data can severely compromise the accuracy of any predictive model. Imputation with the median loan amount or utilizing model-based imputation techniques to estimate these missing values based on other available features is recommended.

Similarly, the "Loan_Amount_Term" variable has 14 missing entries, accounting for 2.3% of the dataset. Since the loan term duration is crucial for assessing an applicant's repayment capacity, the absence of this information can lead to erroneous predictions. Imputing the missing values with the median term duration, given its common values, can help maintain the dataset's integrity.

The "Credit_History" variable, with 50 missing values, poses the most significant challenge, representing 8.1% of the dataset. Credit history is a vital predictor of an applicant’s creditworthiness, and missing data in this variable can skew the model towards inaccurate predictions. Imputing these missing values with the mode or introducing a separate category for missing credit history ensures that the model can handle such cases appropriately.

Variables such as "ApplicantIncome," "CoapplicantIncome," "Property_Area," and "Loan_Status" exhibit no missing values, thereby providing a reliable foundation for the model. These variables can serve as stable predictors in the machine learning process.

### Checking for Outliers using Boxplots

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/0cf59fcf-730a-488a-9786-35aa35e9f6e1)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/c66ab67a-2ba0-49ae-b11e-64d20ce17efa)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/11e818fe-0d09-4915-989c-d81385577290)

The analysis of the provided boxplots for "ApplicantIncome," "CoapplicantIncome," and "LoanAmount" reveals the presence of significant outliers. These outliers are data points that deviate markedly from the other observations and can potentially skew the analysis and modeling processes. Addressing these outliers is crucial for maintaining data quality and ensuring accurate model performance.

The boxplot for "ApplicantIncome" indicates several extreme outliers, with some values exceeding 80,000. Given that the median applicant income is substantially lower, these extreme values can disproportionately influence the mean and standard deviation, leading to skewed results. The presence of these outliers can result in a biased model that overemphasizes the importance of high-income applicants, leading to inaccurate predictions, particularly for those with average or below-average incomes. To mitigate this issue, techniques such as capping the outliers at a certain threshold, transforming the data using logarithmic or square root functions, or employing robust statistical methods that are less sensitive to extreme values should be considered.

Similarly, the boxplot for "CoapplicantIncome" displays significant outliers, with some values extending beyond 40,000. These outliers can also distort the statistical analysis and model predictions, as the majority of coapplicant incomes appear to be much lower. The impact of these outliers is similar to that of the applicant incomes, potentially biasing the model towards high-income coapplicants and reducing its accuracy for the general population. Addressing these outliers through similar techniques as those suggested for applicant incomes is necessary to ensure a balanced and reliable dataset.

The "LoanAmount" variable also exhibits numerous outliers, with values surpassing 600. Given that the median loan amount is significantly lower, these extreme values can skew the data distribution and affect the model's predictive power. Outliers in loan amounts can lead to a model that inaccurately predicts loan eligibility, especially for applicants requesting average loan amounts. To address this, applying transformations, capping the extreme values, or using robust statistical methods will help maintain the integrity of the dataset and enhance the model's performance.

### Checking for class imbalance in the categorical variables using Bargraphs

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/f590f988-ffde-4a04-b6d5-cc47b1c2afe3)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/fef99d8d-6888-4026-806d-87b6581a4fe8)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/be608cb6-41e8-4137-b732-2832806a68c3)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/1b2134fa-1e67-42b8-a0ea-a768bfbac069)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/ef05ab0d-2a46-4259-88c1-195ff4b03c64)

The class distributions for several key categorical variables in our dataset reveal significant insights and potential data quality issues related to class imbalance. For the variable "Self Employment," the majority of applicants (81.4%) are not self-employed, with only 13.4% being self-employed and 5.2% of the data missing. This stark imbalance suggests that the model may become biased towards predicting the "No" category, potentially underperforming in identifying self-employed applicants accurately. Similarly, the "Loan Status" variable exhibits a pronounced imbalance, with 68.7% of loans approved ("Yes") and only 31.3% rejected ("No"). This imbalance poses a risk of the model favoring loan approvals, which could result in overlooking cases that should be denied.

In contrast, the "Property Area" variable displays a relatively balanced distribution, with 29.2% of applicants from rural areas, 37.9% from semiurban areas, and 32.9% from urban areas. Although there is a slight overrepresentation of semiurban areas, this imbalance is not as pronounced and is less likely to significantly affect model performance.

Addressing class imbalance is crucial for developing robust and reliable machine learning models. Techniques such as resampling (both oversampling the minority class and undersampling the majority class), applying cost-sensitive learning algorithms, and using appropriate evaluation metrics like precision, recall, F1-score, and the area under the precision-recall curve are essential. Additionally, data augmentation methods and creating more representative features can further enhance the model's ability to handle imbalanced datasets.

### Checking for duplicates

There was only one duplicated row 

### Data in different scales

The examination of the quantitative variables in the loan dataset reveals significant data quality issues related to differing scales. These variables—"ApplicantIncome," "CoapplicantIncome," and "LoanAmount"—exist on markedly different scales, which can introduce challenges in data consistency and model performance.

"ApplicantIncome" is recorded in the range of hundreds to tens of thousands, with values spanning from 150 to 81,000. In contrast, "CoapplicantIncome" varies from 0 to 41,667, and "LoanAmount" ranges from 9 to 700. The disparity in scales between these variables can pose significant challenges for machine learning algorithms, which often assume that input features are on a comparable scale. When features exist on vastly different scales, models may become biased towards higher magnitude values, leading to suboptimal performance and inaccurate predictions.

To address this issue, it is essential to apply feature scaling techniques to normalize the data. Standardization, which transforms the data to have a mean of zero and a standard deviation of one, is one effective approach. This technique ensures that each variable contributes equally to the model, regardless of its original scale. Alternatively, normalization, which scales the data to a fixed range (typically 0 to 1), can also be employed to achieve similar effects.

For instance, standardizing "ApplicantIncome," "CoapplicantIncome," and "LoanAmount" will transform these variables into comparable units, thereby eliminating the biases introduced by differing scales. This approach ensures that the machine learning model can appropriately weigh each feature, improving its overall performance and accuracy.

### Numerical Data Skewness

Data skewness is a significant data quality issue that can adversely affect the performance and reliability of machine learning models. The provided boxplots and base table for "ApplicantIncome," "CoapplicantIncome," and "LoanAmount" reveal substantial skewness in these quantitative variables, which can lead to biased models and inaccurate predictions.

The boxplot for "ApplicantIncome" demonstrates a pronounced right-skewed distribution, characterized by a long tail extending towards higher income values. This indicates that while the majority of applicants have relatively low to moderate incomes, a small number of applicants report exceptionally high incomes. The base table further supports this observation, showing a mean that is considerably higher than the median, reflecting the influence of these high-income outliers. This skewness can result in a model that is disproportionately influenced by the high-income data points, potentially leading to biased predictions that do not accurately represent the broader applicant population.

Similarly, "CoapplicantIncome" exhibits a right-skewed distribution with several extreme values. The boxplot highlights the presence of high-income outliers among coapplicants, while the base table shows a mean income significantly greater than the median. This skewness can cause the model to overestimate the importance of coapplicant income, particularly those at the higher end of the distribution, thereby affecting the model’s generalizability and accuracy.

The "LoanAmount" variable also shows a right-skewed distribution. The boxplot indicates a concentration of loan amounts at the lower end, with a few significantly larger loan amounts extending the tail. The base table reveals a mean loan amount that exceeds the median, further illustrating the impact of these outliers. Such skewness can bias the model towards higher loan amounts, potentially misrepresenting the typical loan applicant.

To address skewedness and its impact on model performance, it is essential to apply data transformation techniques. Logarithmic or square root transformations can help normalize the distributions, reducing the influence of extreme values and bringing the data closer to a normal distribution. This adjustment ensures that the model treats all data points more equitably, enhancing the robustness and accuracy of predictions.


# Step 3: Data Preparation

Having identified and discussed the data quality issues, we now proceed to the crucial stage of data preparation for modeling. This phase involves comprehensive data preprocessing to ensure the dataset is clean, balanced, and ready for analysis. The preprocessing steps will be tailored to the specific requirements of the following machine learning classification algorithms that will be explored for this project:

    Decision Tree Classifier
    K-Nearest Neighbors (K-NN)
    Support Vector Machines
    Logistic Regression

Our goal is to select the best-performing model from these algorithms based on their predictive accuracy and robustness.

### Removing Duplicates

The dataset was examined for duplicate observations, and any duplicates found were removed. This step is crucial to ensure the integrity of the data, as duplicates can distort the analysis and lead to misleading conclusions by artificially inflating the sample size.

#### Handling Missing Values

Categorical missing values were imputed using the mode, while numerical missing values were imputed using the median. This approach ensures that the most frequent category replaces missing categorical data, maintaining the mode's integrity, and the median imputation prevents the influence of outliers on numerical data. These imputation methods preserve the dataset's distribution and prevent the loss of valuable information, making the data ready for algorithms such as Decision Trees, which can handle outliers and varying scales without requiring normalization or standardization.

### Handling Outliers Using the Boxplot Method

Outliers were identified and handled using the boxplot method. Boxplots are a graphical representation of data that highlight the distribution and identify extreme values or outliers. In this process, outliers were detected as points lying outside 1.5 times the interquartile range (IQR) from the first and third quartiles. Handling outliers is essential because they can skew the results of the analysis and negatively impact the performance of machine learning models. By addressing outliers—either by capping their values at acceptable limits or by treating them separately—we ensure that the data is more representative of the typical values and improve the model's robustness and accuracy.

#### Standardization of Quantitative Variables

Standardization was applied to transform the quantitative variables to the same scale. This step was essential for algorithms like K-Nearest Neighbors (K-NN) and Support Vector Machines (SVM), which are sensitive to the ranges of data. Standardizing the variables ensures that each feature contributes equally to the model, preventing features with larger scales from dominating the analysis.

#### One-Hot Encoding for Categorical Variables

One-hot encoding was used to convert categorical variables into dummy variables. This transformation is necessary for K-Nearest Neighbors, Support Vector Machines, and Logistic Regression algorithms, which cannot process categorical data directly. By creating binary columns for each category, one-hot encoding allows these algorithms to interpret and utilize categorical information effectively.

# Step 4: Modeling

The dataset was partitioned into training and testing sets with an 80:20 split ratio to ensure a robust evaluation of the machine learning models. The target variable, originally represented as 'Y' and 'N', was converted to binary values of 1 and 0, respectively, to facilitate compatibility with the classification algorithms. This conversion allows the models to effectively learn and predict the loan eligibility status based on the provided feature

# Step 5: Model Performance and Evaluation

The performance of the four machine learning models—Decision Tree, K-Nearest Neighbors (K-NN), Support Vector Machines (SVM), and Logistic Regression—was evaluated using accuracy and Cohen's kappa (kap) metrics. These metrics provide insights into how well each model performs in predicting loan eligibility.

### Decision Tree

Accuracy: 0.782581
Kappa: 0.4183461
The Decision Tree classifier achieved an accuracy of approximately 78.3%, indicating that it correctly predicted the loan eligibility status for 78.3% of the instances. The kappa statistic, which measures the agreement between predicted and actual classifications while accounting for chance, is 0.418. This suggests moderate agreement beyond chance.

### K-Nearest Neighbors (K-NN)

Accuracy: 0.750000
Kappa: 0.3627321
The K-NN classifier achieved an accuracy of 75%, slightly lower than the Decision Tree. The kappa value of 0.363 indicates a fair agreement beyond chance. The performance of K-NN is slightly lower than the Decision Tree in terms of both metrics.

### Support Vector Machines (SVM)

Accuracy: 0.8145161
Kappa: 0.5045170
The SVM classifier outperformed the Decision Tree and K-NN with an accuracy of approximately 81.5%. The kappa statistic of 0.504 indicates moderate agreement beyond chance, higher than that of the Decision Tree and K-NN. This suggests that SVM provides a better balance between precision and recall.

### Logistic Regression

Accuracy: 0.8145161
Kappa: 0.5123119
Logistic Regression achieved the same accuracy as SVM, approximately 81.5%. However, it has a slightly higher kappa value of 0.512, indicating moderate agreement beyond chance. This suggests that Logistic Regression slightly outperforms SVM in terms of consistency between predicted and actual classifications.

### Selection of the Best Model

Based on the evaluation metrics, Logistic Regression and SVM are the top-performing models, both achieving an accuracy of 81.5%. However, Logistic Regression has a marginally higher kappa value (0.512) compared to SVM (0.504), indicating a slightly better performance in terms of agreement between predicted and actual classifications.

Therefore, Logistic Regression is selected as the best model for predicting loan eligibility in this dataset. Its balance of high accuracy and kappa value demonstrates its robustness and reliability in handling this classification task. The slight edge in kappa value suggests that it offers a better balance between precision and recall, making it a more reliable choice for real-world application.



















