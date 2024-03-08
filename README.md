# CO2_emissions_PROJECT

## Project:

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

# Methods and Technologies used:

1.Exploratory Data Analysis

2.Data visualisation

3.Machine learning

# Technologies:

1.Python

2.Pandas, NumPy

3.Scikit-learn

4.Matplotlib

5.Seaborn

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

- support  Vector Machine:
  
SVM is also a versatile learning algo able to perform linear and non-linear classification, regression and even outlier detection. Here Linear SVM is used.

# Metrics used for model evaluation:

1. RMSE(Root Mean Squared Error):
 
RMSE is the standard deviation of the prediction errors called residuals. Residuals point to the randomness of data points from the regression line. RMSE tells us how concentrated the data is around the regression line.

2. MAE(Mean Absolute Error):

MAE measures the absolute distance between the real data and predicted data.

3. R2 score:

R-Square is another statistical measure which indicates how well the regression predictions approximate the real data points.
