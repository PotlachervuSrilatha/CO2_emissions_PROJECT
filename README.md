# Project:

# CO2_Emissions_PROJECT

#### Business Objective:

The fundamental goal here is to model the CO2 emissions as a function of several car engine features.

## Data Set Details: 

The file contains the data for this example. Here the number of variables (columns) is 12, and the number of instances (rows) is 7385. In that way, this problem has the 12 following variables:

- make, car brand under study.
- model, the specific model of the car.
- vehicle_class, car body type of the car.
- engine_size, size of the car engine, in Liters.
- cylinders, number of cylinders.
- transmission, "A" for`Automatic', "AM" for ``Automated manual', "AS" for 'Automatic with select shift', "AV" for 'Continuously variable', "M" for 'Manual'.
- fuel_type, "X" for 'Regular gasoline', "Z" for 'Premium gasoline', "D" for 'Diesel', "E" for 'Ethanol (E85)', "N" for 'Natural gas'.
- fuel_consumption_city, City fuel consumption ratings, in liters per 100 kilometers.
- fuel_consumption_hwy, Highway fuel consumption ratings, in liters per 100 kilometers.
- fuel_consumption_comb(l/100km), the combined fuel consumption rating (55% city, 45% highway), in L/100 km.
- fuel_consumption_comb(mpg), the combined fuel consumption rating (55% city, 45% highway), in miles per gallon (mpg).
- co2_emissions, the tailpipe emissions of carbon dioxide for combined city and highway driving, in grams per kilometer.

# Table of Contents
# Import Libraries
- Set Options
- Read Data
- Exploratory Data Analysis
  
- 4.1 - Preparing the Dataset

4.1.1 - Data Dimension

4.1.2 - Data Types

4.1.3 - Missing Values

4.1.4 - Duplicate Data

- 4.2 - Understanding the Dataset

4.2.1 - Summary Statistics

4.2.2 - Correlation

4.2.3 - Analyze Categorical Variables

4.2.4 - Anaylze Target Variable


4.2.5 - Analyze Relationship Between Target and Independent Variables
4.2.6 - Feature Engineering

- Data Pre-Processing
  
5.1 - Outliers

5.1.1 - Discovery of Outliers

5.1.2 - Removal of Outliers

5.1.3 - Rechecking of Correlation

5.2 - Categorical Encoding

- Building Multiple Linear Regression Models
  
6.1 - Multiple Linear Regression - Basic Model

6.2 - Feature Transformation

6.3 - Feature Scaling

6.4 - Multiple Linear Regression - Full Model - After Feature Scaling

6.5 - Assumptions Before Multiple Linear Regression Model

6.5.1 - Assumption #1: If Target Variable is Numeric

6.5.2 - Assumption #2: Presence of Multi-Collinearity

6.6 - Multiple Linear Regression - Full Model - After PCA

6.7 - Feature Selection

6.7.1 - Forward Selection

6.7.2 - Backward Elimination

- 6.8 - Multiple Linear Regression - Full Model - After Feature Selection
  
6.9 - Assumptions After Multiple Linear Regression Model

6.9.1 - Assumption #1: Linear Relationship Between Dependent and Independent Variable

6.9.2 - Assumption #2: Checking for Autocorrelation

6.9.3 - Assumption #3: Checking for Heterskedacity

6.9.4 - Assumption #4: Test for Normality

6.9.4.1 - Q-Q Plot

6.9.4.2 - Shapiro Wilk Test

- Model Evaluation
  
7.1 - Measures of Variation

7.2 - Inferences about Intercept and Slope

7.3 - Confidence Interval for Intercept and Slope

7.4 - Compare Regression Results

- Model Performance
  
8.1 - Mean Square Error(MSE)

8.2 - Root Mean Squared Error(RMSE)

8.3 - Mean Absolute Error(MAE)

8.4 - Mean Absolute Percentage Error(MAPE)

8.5 - Resultant Table

- Model Optimization
  
9.1 - Bias

9.2 - Variance

9.3 - Model Validation

9.3.1 - Cross Validation


9.3.2 - Leave One Out Cross Validation(LOOCV)

9.4 - Gradient Descent

9.5 - Regularization

9.5.1 - Ridge Regression Model

9.5.2 - Lasso Regression Model

9.5.3 - Elastic Net Regression Model

9.5.4 - Grid Search CV

- Displaying Score Summary
  
- Conclusion

# Data Description

- Model

4WD/4X4 = Four-wheel drive

AWD = All-wheel drive

FFV = Flexible-fuel vehicle

SWB = Short wheelbase

LWB = Long wheelbase

EWB = Extended wheelbase

- Transmission

A = Automatic

AM = Automated manual

AS = Automatic with select shift

AV = Continuously variable

M = Manual

3 - 10 = Number of gears

- Fuel type

X = Regular gasoline

Z = Premium gasoline

D = Diesel

E = Ethanol (E85)

N = Natural gas

- Fuel Consumption

City and highway fuel consumption ratings are shown in litres per 100 kilometres (L/100 km) - the combined rating (55% city, 45% hwy) is shown in L/100 km and in miles per gallon (mpg).



# Technologies:

1.Python ( 3.8 version)

2.Pandas, NumPy

3.Scikit-learn

4.Matplotlib

5.Seaborn

6.Flask

7.Pickle

# Machine learning algos used:

- linear regression:
  
The most basic regression algorithm which make predictions by simply computing weighted sum of input features adding a bias term.

- Lasso Regression:
  
Least Absolute Shrinkage and Selection Operator Regression(Lasso) is a regularized version of linear regression which adds a regularization term to the cost function using l1 norm or Manhattan norm.

- Ridge Regression:

Similar to lasso regression but uses l2 norm or Eucledian norm of the weight vector.

- Decision Tree :
  
A versatile learning algorithm that can perform regression, classification, multi-ouputs tasks even on complex data sets using tree structure(root and leaf nodes).

- Random Forest:

Random forest is an ensemble of decision trees which introduces extra randomness(that is it searches for best feature among a random subset of fatures) when growing trees instead of searching for very best feature when splitting a node.

- support  Vector Regressor:
  
SVM is also a versatile learning algo able to perform linear and non-linear classification, regression and even outlier detection. Here Linear SVM is used.

# Metrics used for model evaluation:

1. RMSE(Root Mean Squared Error):
 
RMSE is the standard deviation of the prediction errors called residuals. Residuals point to the randomness of data points from the regression line. RMSE tells us how concentrated the data is around the regression line.

2. MAE(Mean Absolute Error):

MAE measures the absolute distance between the real data and predicted data.

3. R2 score:

R-Square is another statistical measure which indicates how well the regression predictions approximate the real data points.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Deployment

We used Random Forest for final deployment :
![image](https://github.com/PotlachervuSrilatha/CO2_emissions_PROJECT/assets/97737090/76e40ddc-459d-4056-bd9b-6baf56bcb446)
![image](https://github.com/PotlachervuSrilatha/CO2_emissions_PROJECT/assets/97737090/a6d30e52-4316-46bb-8e59-5b5bc05f39ee)

