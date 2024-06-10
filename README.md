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

#### Base Table Creation

The first step in understanding our data was to create a base table that consolidates all relevant variables. This base table includes attributes such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others.

Observing the first 6 rows of the data gives a high level view of the variables 

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/f019d59c-d744-41db-be90-c653b5230e6d)

The dataset has 614 Observations and 13 variables. The Loan_ID id variable is dropped as it adds no significant meaning to the machine learning performance.The target variable is the Loan_Status that states if a client was provided with a home loan or not.

#### Checking the number of missing values per variable

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/ee25ad0e-1bca-4ed6-ac48-87e889cdfc95)

### Checking for Outliers using Boxplots

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/0cf59fcf-730a-488a-9786-35aa35e9f6e1)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/c66ab67a-2ba0-49ae-b11e-64d20ce17efa)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/dae0b4cb-ffed-49f6-865a-3146c0e52052)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/11e818fe-0d09-4915-989c-d81385577290)

### Checking for class imbalance in the categorical variables using Bargraphs

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/f590f988-ffde-4a04-b6d5-cc47b1c2afe3)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/fef99d8d-6888-4026-806d-87b6581a4fe8)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/be608cb6-41e8-4137-b732-2832806a68c3)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/1b2134fa-1e67-42b8-a0ea-a768bfbac069)

![image](https://github.com/HlabirwaM/Machine-Learning-Model-For-Optimising-Banking-Campaign-Strategy/assets/158508458/ef05ab0d-2a46-4259-88c1-195ff4b03c64)

The class distributions for several key categorical variables in our dataset reveal significant insights and potential data quality issues related to class imbalance. For the variable "Self Employment," the majority of applicants (81.4%) are not self-employed, with only 13.4% being self-employed and 5.2% of the data missing. This stark imbalance suggests that the model may become biased towards predicting the "No" category, potentially underperforming in identifying self-employed applicants accurately. Similarly, the "Loan Status" variable exhibits a pronounced imbalance, with 68.7% of loans approved ("Yes") and only 31.3% rejected ("No"). This imbalance poses a risk of the model favoring loan approvals, which could result in overlooking cases that should be denied.

In contrast, the "Property Area" variable displays a relatively balanced distribution, with 29.2% of applicants from rural areas, 37.9% from semiurban areas, and 32.9% from urban areas. Although there is a slight overrepresentation of semiurban areas, this imbalance is not as pronounced and is less likely to significantly affect model performance.

Addressing class imbalance is crucial for developing robust and reliable machine learning models. Techniques such as resampling (both oversampling the minority class and undersampling the majority class), applying cost-sensitive learning algorithms, and using appropriate evaluation metrics like precision, recall, F1-score, and the area under the precision-recall curve are essential. Additionally, data augmentation methods and creating more representative features can further enhance the model's ability to handle imbalanced datasets.




