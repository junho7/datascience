
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Read the data set into a pandas DataFrame
churn = pd.read_csv('datascience/churn/churn.csv', sep=',', header=0)

churn.head()

# clean column names
churn.columns = [heading.lower() for heading in churn.columns.str.replace(' ', '_').str.replace("\'", "").str.strip('?')]

# convert churn column from categorical to number
churn['churn01'] = np.where(churn['churn'] == 'True.', 1., 0.)
print(churn.head())
print(churn.describe())

# Calculate descriptive statistics for grouped data
print(churn.groupby(['churn'])[['day_charge', 'eve_charge', 'night_charge', 'intl_charge', 'account_length', 'custserv_calls']].agg(['count', 'mean', 'std']))

# Specify different statistics for different variables
print(churn.groupby(['churn']).agg({'day_charge' : ['mean', 'std'], 
				'eve_charge' : ['mean', 'std'],
				'night_charge' : ['mean', 'std'],
				'intl_charge' : ['mean', 'std'],
				'account_length' : ['count', 'min', 'max'],
				'custserv_calls' : ['count', 'min', 'max']}))

# Create total_charges, split it into 5 groups, and
# calculate statistics for each of the groups
churn['total_charges'] = churn['day_charge'] + churn['eve_charge'] + \
						 churn['night_charge'] + churn['intl_charge']
# cut produces equal-width bins					
# ex. 100 -> 20, 40, 60 ,80, 100
factor_cut = pd.cut(churn.total_charges, 5, precision=2)

def get_stats(group):
	return {'min' : group.min(), 'max' : group.max(),
			'count' : group.count(), 'mean' : group.mean(),
			'std' : group.std()}

grouped = churn.custserv_calls.groupby(factor_cut)

# unstack: rows -> columns
print(grouped.apply(get_stats).unstack())

# Split account_length into quantiles and
# calculate statistics for each of the quantiles
# qcut produces equal-sized bins (i.e. quantiles)
factor_qcut = pd.qcut(churn.account_length, [0., 0.25, 0.5, 0.75, 1.])
grouped = churn.custserv_calls.groupby(factor_qcut)
print(grouped.apply(get_stats).unstack())

# Create binary/dummy indicator variables for intl_plan and vmail_plan
# and join them with the churn column in a new DataFrame
intl_dummies = pd.get_dummies(churn['intl_plan'], prefix='intl_plan')
vmail_dummies = pd.get_dummies(churn['vmail_plan'], prefix='vmail_plan')
churn_with_dummies = churn[['churn']].join([intl_dummies, vmail_dummies])
print(churn_with_dummies.head())

# Split total_charges into quartiles, create binary indicator variables
# for each of the quartiles, and add them to the churn DataFrame
qcut_names = ['1st_quartile', '2nd_quartile', '3rd_quartile', '4th_quartile']
total_charges_quartiles = pd.qcut(churn.total_charges, 4, labels=qcut_names)
dummies = pd.get_dummies(total_charges_quartiles, prefix='total_charges')
churn_with_dummies = churn.join(dummies)
print(churn_with_dummies.head())

# Create pivot tables
# margins=True provides 'all' column and rows
print(churn.pivot_table(['total_charges'], index=['churn', 'custserv_calls']))
print(churn.pivot_table(['total_charges'], index=['churn'], columns=['custserv_calls']))
print(churn.pivot_table(['total_charges'], index=['custserv_calls'], columns=['churn'], \
						aggfunc='mean', fill_value='NaN', margins=True))

# Fit a logistic regression model
dependent_variable = churn['churn01']
independent_variables = churn[['account_length', 'custserv_calls', 'total_charges']]
# constant is intercept. if prepend=True, column will be first column
independent_variables_with_constant = sm.add_constant(independent_variables, prepend=True)
logit_model = sm.Logit(dependent_variable, independent_variables_with_constant).fit()
#logit_model = smf.glm(output_variable, input_variables, family=sm.families.Binomial()).fit()
print(logit_model.summary())
print("\nQuantities you can extract from the result:\n%s" % dir(logit_model))
print("\nCoefficients:\n%s" % logit_model.params)
print("\nCoefficient Std Errors:\n%s" % logit_model.bse)
#logit_marginal_effects = logit_model.get_margeff(method='dydx', at='overall')
#print(logit_marginal_effects.summary())

print("\ninvlogit(-7.2205 + 0.0012*mean(account_length) + 0.4443*mean(custserv_calls) + 0.0729*mean(total_charges))")

#Interpreting the regression coefficients for a logistic regression isn't as straightforward
# as it is for a linear regression.
# because the inverse logistic function is curved,
# which means the expected difference in the dependent variable for a one-unit change
# in an independenct variable isn't constant
# one way is , as with linear regression, the coefficient for the intercept
# represents the probability of success when all of the independent variables are zero
# Sometimes zero value don't make sense, so alternative is to evaluate the function
# with all of the independent variables set to their mean values

# transforms the continuous predicted values from the linear model into probabilities between 0 and 1
def inverse_logit(model_formula):
	from math import exp
	return (1.0 / (1.0 + exp(-model_formula)))*100.0

#
at_means = float(logit_model.params[0]) + \
	float(logit_model.params[1])*float(churn['account_length'].mean()) + \
	float(logit_model.params[2])*float(churn['custserv_calls'].mean()) + \
	float(logit_model.params[3])*float(churn['total_charges'].mean())

print(churn['account_length'].mean())
print(churn['custserv_calls'].mean())
print(churn['total_charges'].mean())
# at_means:  -2.068
print(at_means)
# the inverse logit of -2.068 is 0.112.
# probability of person churning whose account length, customer service calls, and 
# total charges are equal to the avg values for these variables is 11.2%
print("Probability of churn when independent variables are at their mean values: %.2f" % inverse_logit(at_means))

#evaluate the impact of a one-unit change in the number of customer service calls,
# close to this variable's mean value, on the probability of churning
cust_serv_mean = float(logit_model.params[0]) + \
	float(logit_model.params[1])*float(churn['account_length'].mean()) + \
	float(logit_model.params[2])*float(churn['custserv_calls'].mean()) + \
	float(logit_model.params[3])*float(churn['total_charges'].mean())
		
cust_serv_mean_minus_one = float(logit_model.params[0]) + \
		float(logit_model.params[1])*float(churn['account_length'].mean()) + \
		float(logit_model.params[2])*float(churn['custserv_calls'].mean()-1.0) + \
		float(logit_model.params[3])*float(churn['total_charges'].mean())

print(cust_serv_mean)
print(churn['custserv_calls'].mean()-1.0)
print(cust_serv_mean_minus_one)
print("Probability of churn when account length changes by 1: %.2f" % (inverse_logit(cust_serv_mean) - inverse_logit(cust_serv_mean_minus_one)))

# Predict churn for "new" observations
# create 10 new observations from the first 10 observations in the churn dataset
new_observations = churn.ix[churn.index.isin(range(10)), independent_variables.columns]
new_observations_with_constant = sm.add_constant(new_observations, prepend=True)

#predict probability of churn based on the new observations' account characteristics
y_predicted = logit_model.predict(new_observations_with_constant)
y_predicted_rounded = [round(score, 2) for score in y_predicted]
print(y_predicted_rounded)

# Fit a logistic regression model
output_variable = churn['churn01']
vars_to_keep = churn[['account_length', 'custserv_calls', 'total_charges']]
inputs_standardized = (vars_to_keep - vars_to_keep.mean()) / vars_to_keep.std()
input_variables = sm.add_constant(inputs_standardized, prepend=False)
logit_model = sm.Logit(output_variable, input_variables).fit()
#logit_model = smf.glm(output_variable, input_variables, family=sm.families.Binomial()).fit()
print(logit_model.summary())
print(logit_model.params)
print(logit_model.bse)
#logit_marginal_effects = logit_model.get_margeff(method='dydx', at='overall')
#print(logit_marginal_effects.summary())

# Predict output value for a new observation based on its mean standardized input values
input_variables = [0., 0., 0., 1.]
predicted_value = logit_model.predict(input_variables)
print (("Predicted value: %.5f") % predicted_value)

