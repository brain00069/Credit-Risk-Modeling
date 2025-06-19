# Credit-Risk-Modeling
## Introduction

This project focuses on developing a predictive credit risk model to assess the likelihood of customers defaulting on auto loans, based on various borrower and loan characteristics. By analyzing an auto loan portfolio, we aim to identify patterns and key indicators that influence default risk. This initiative is particularly relevant as financial institutions seek robust, data-driven solutions to mitigate risks, optimize lending decisions, and enhance overall portfolio performance in an increasingly data-centric industry.

Credit risk modeling is a critical tool for lenders, enabling them to evaluate potential risks and allocate resources effectively. By predicting defaults, institutions can proactively manage their portfolios, minimize losses, and maximize returns. For this project, we aim to explore multiple machine learning methodologies used in real-world applications, offering insights into both their predictive capabilities and practical implementation.

The data for this project is sourced from a publicly available dataset on Kaggle, comprising detailed training and testing data. With approximately 40 variables, including a classification column for loan status, the dataset offers rich information for identifying significant predictors of default. However, we will prioritize selecting the most impactful features to prevent overfitting and enhance model reliability. Any necessary preprocessing, including adjustments to the testing set size, will be addressed during the data preparation stage.

This project aims to not only deliver an accurate predictive model but also provide valuable insights into the process of credit risk assessment, emphasizing practical applications in financial decision-making.

## Source of Data
### Data Summary
This dataset contains 40 columns, including information relevant to assessing borrowers' repayment abilities. There is a combination of various data types in the initial dataset, highlighting some initial obstacles that our group must overcome in the cleaning process. The objective is to build a predictive model to determine the likelihood of loan defaults. The data focuses on analyzing and identifying the factors contributing to repayment behavior, aiding in risk assessment and improving loan decisions.

### Description
Below is a detailed description of each of the features in our dataset:
| Variable                    | Description                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| `ID`                        | Client Loan application ID                                                                    |
| `Client_Income`             | Client income in $                                                                            |
| `Car_Owned`                 | Whether the client owns a car (0 = No, 1 = Yes)                                               |
| `Bike_Owned`                | Whether the client owns a bike (0 = No, 1 = Yes)                                              |
| `Active_Loan`              | Whether the client has any active loans (0 = No, 1 = Yes)                                     |
| `House_Own`                 | Whether the client owns a house (0 = No, 1 = Yes)                                             |
| `Child_Count`               | Number of children the client has                                                             |
| `Credit_Amount`             | Credit amount of the loan in $                                                                |
| `Loan_Annuity`              | Loan annuity in $                                                                             |
| `Accompany_Client`          | Who accompanied the client during the loan application                                        |
| `Client_Income_Type`        | Client’s income type                                                                          |
| `Client_Education`          | Client’s highest level of education                                                           |
| `Client_Marital_Status`     | Marital status (`D` = Divorced, `S` = Single, `M` = Married, `W` = Widowed)                   |
| `Client_Gender`             | Gender of the client                                                                          |
| `Loan_Contract_Type`        | Loan type (`CL` = Cash Loan, `RL` = Revolving Loan)                                           |
| `Client_Housing_Type`       | Client’s housing situation                                                                    |
| `Population_Region_Relative`| Relative population density of client’s region                                                |
| `Age_Days`                  | Age of client in days at time of application                                                  |
| `Employed_Days`             | Days since the client started earning                                                         |
| `Registration_Days`         | Days since the client changed their registration                                              |
| `ID_Days`                   | Days since the client changed identity document                                               |
| `Own_House_Age`             | Age of client’s house in years                                                                |
| `Mobile_Tag`                | Whether mobile number is provided (1 = Yes, 0 = No)                                           |
| `Homephone_Tag`             | Whether home phone number is provided (1 = Yes, 0 = No)                                       |
| `Workphone_Working`         | Whether work phone number is reachable (1 = Yes, 0 = No)                                      |
| `Client_Occupation`         | Type of client occupation                                                                     |
| `Client_Family_Members`     | Number of family members                                                                      |
| `Cleint_City_Rating`        | Client city rating (1 = Average, 2 = Good, 3 = Best)                                          |
| `Application_Process_Day`   | Day of week of application (`0`=Sun ... `6`=Sat)                                              |
| `Application_Process_Hour`  | Hour of the day the application was submitted                                                 |
| `Client_Permanent_Match_Tag`| Whether contact address matches permanent address (1 = Match, 0 = No Match)                  |
| `Client_Contact_Work_Tag`   | Whether work address matches contact address (1 = Match, 0 = No Match)                       |
| `Type_Organization`         | Type of organization the client works for                                                    |
| `Score_Source_1`            | External normalized credit score #1                                                           |
| `Score_Source_2`            | External normalized credit score #2                                                           |
| `Score_Source_3`            | External normalized credit score #3                                                           |
| `Social_Circle_Default`     | Number of defaults by friends/family in last 60 days                                          |
| `Phone_Change`              | Days since client changed phone number                                                        |
| `Credit_Bureau`             | Number of credit inquiries in the past year                                                   |
| `Default`                   | Loan default status (1 = Defaulted, 0 = Not Defaulted)                                        |

## Scope of Analysis
To ensure the data was well-prepared for modeling and to improve the robustness of predictions, we conducted the analysis in three stages:

1. Benchmarking: Established baseline performance using Target attribute proportions and Logistic Regression without extensive preprocessing. This step provided an initial understanding of the dataset and its predictive potential.

2. Standardization and Outlier Removal: Standardized the numeric features to bring them to a similar scale, improving model performance for algorithms sensitive to feature scaling. Outliers were identified and addressed to reduce their undue influence on the models.

3. SMOTE (Synthetic Minority Oversampling Technique) Application: Addressed class imbalance by oversampling the minority class to ensure the models were not biased toward predicting the majority class. This step improved recall and balanced overall performance.

## Models Explored
1. Logistic Regression
Logistic regression is particularly advantageous for classification-prediction problems; it is a simple and interpretable model that can effectively predict the probability of a binary outcome based on a variety of input features.

2. Naive Bayes
Naive Bayes is fast, easy to implement, and works well with high-dimensional data. It also assumes feature independence, which simplifies computations.

3. Decision Tree 
Decision trees are intuitive, non-parametric models that can capture non-linear relationships and interactions among features. They also provide interpretable decision rules, which is valuable in financial applications. While decision trees performed better than logistic regression and Naive Bayes, they exhibited a tendency to overfit the training data, resulting in reduced generalizability. Additionally, the predictions were overly sensitive to small changes in the dataset, which affected consistency.

## Model Recommendation:
After comparing the results from benchmarks and modeling techniques, it looks like the Naive Bayes algorithm (with SMOTE) - with accuracy measures of about 77% on train and 74% on test samples- is performing relatively better. Additionally, the model created the highest stratified accuracy for defaults as compared to other models. This model would be recommended for this problem because of a relatively high overall accuracy and its ability to correctly categorize default status as compared to other models explored, which is vital for businesses to make risk decisions. However, our models are not to the standard at which a business could reasonably rely on, therefore the overall recommendation is continued exploration and adjustments until the model reaches adequate performance.

## Learning
### Key Insights and Takeaways from Data Mining Analysis
1. Model Types
Depending on the goal and the data available, certain models are better than others.

2. Cleaning Data
Real-world datasets are often messy and require significant preprocessing before they can be used effectively in machine learning. For this project, we addressed common issues such as missing values, outliers, and inconsistencies in the dataset.

3. Preprocessing
Preprocessing is a critical step in achieving the best results, as it allows us to enhance data quality and ensure compatibility with various algorithms. In this project, preprocessing involved:

• Feature Engineering: Transforming raw data into meaningful variables, including converting categorical data into dummy variables and scaling numerical features.

• Variable Testing: Experimenting with different feature subsets to identify the most impactful predictors for the model.

• Optimization of Preprocessing Techniques: Applying techniques such as outlier removal, standardization, and SMOTE to balance the dataset and improve overall model performance.

### Translating Data Insights into Actionable Strategies for Business Managers

The loan default classification model provides better insights based on customer characteristics, allowing businesses to better manage potential risks and maximize profits. By avoiding high-risk customers and optimizing offers for low-risk customers, businesses can improve overall operational efficiency. However, every model has its limitations and flaws, so it is crucial to make adjustments based on the situation.

The expected loss for incorrectly categorizing a client is significant for the business. If the business incorrectly classifies a person as a defaulter when they truly would be a non-defaulter, they would lose nothing; they simply would not even give out the loan in the first place, hence no financial loss. Alternatively, there would be a loss of potential business. On the other hand, if the business gives out a loan to someone who is truly a defaulter, then the business would be out the money the defaulter should have paid (the payment amount, or even the entire loan). Therefore, businesses should look for a model that can recognise both classifications and give a decent accuracy on both: a model with high recall and stratified accuracies. That is why processes like SMOTE are important to balance the classification proportions to train the models correctly. 
