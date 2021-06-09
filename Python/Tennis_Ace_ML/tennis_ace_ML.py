import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')

# perform exploratory analysis here:
#print(df.columns)
#print(df.TotalPointsWon)


# Some scatter plots to see some things
#ax1 = plt.subplot(2,2,1)
#plt.scatter(df[['TotalPointsWon']], df[['Wins']])
#plt.xlabel('Total Points')
#plt.ylabel('Wins')

#ax2 = plt.subplot(2,2,2)
#plt.scatter(df[['FirstServePointsWon']], df[['Wins']])
#plt.xlabel('First Serve Points Won')
#plt.ylabel('Wins')

#ax3 = plt.subplot(2,2,3)
#plt.scatter(df[['Aces']], df[['Wins']])
#plt.xlabel('Aces')
#plt.ylabel('Wins')

#ax4 = plt.subplot(2,2,4)
#plt.scatter(df[['BreakPointsSaved']], df[['Wins']])
#plt.xlabel('Break Points Saved')
#plt.ylabel('Wins')

#plt.show()

## perform single feature linear regressions here:

X1 = df[['TotalPointsWon']] # choosing the easiest feature - total points won
y1 = df[['Wins']] # to predict wins

# Instantiate a linear regression object for single variable
s_lin = LinearRegression()

# Split the data into sections to train the model (80%) and test the model (20%)
x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=3)

# Fit a model to the training data and save it
single_model = s_lin.fit(x_train, y_train)

# Make some predictions based on the test x data
win_predict = s_lin.predict(x_test)

# Print out the results
print('Coefficients')
print(s_lin.coef_)

print("Train score:")
print(s_lin.score(x_train, y_train))

print("Test score:")
print(s_lin.score(x_test, y_test))

fig1 = plt.figure()
plt.scatter(y_test, win_predict)
plt.show()
plt.close()

######### perform two feature linear regressions here ############

X2 = df[['TotalPointsWon', 'Aces']] # choosing the easiest feature - total points won
y2 = df[['Wins']] # to predict wins

# Instantiate a linear regression object
dlr = LinearRegression()

# Split the data into sections to train the model (80%) and test the model (20%)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=3)

# Fit a model to the training data
double_model = dlr.fit(x_train2, y_train2)

# Make some predictions based on the test x data
win_predict2 = dlr.predict(x_test2)

# Print out the results
print('Coefficients')
print(dlr.coef_)

print("Train score:")
print(dlr.score(x_train2, y_train2))

print("Test score:")
print(dlr.score(x_test2, y_test2))

fig2 = plt.figure()
plt.scatter(y_test2, win_predict2)
plt.show()
plt.close()

########## perform multiple feature linear regressions here #############

# I started by selecting all of the stats and then eliminated them
# one-by-one and evaluating the train and test scores
# I was able to remove 
        # 'FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon',
        # 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces'
        # 'BreakPointsConverted', 'BreakPointsSaved','DoubleFaults',
        # 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed', 
        # 'ServiceGamesWon','TotalPointsWon', 'TotalServicePointsWon'

X3 = df[['BreakPointsFaced','BreakPointsOpportunities','ReturnGamesPlayed']]
y3 = df[['Wins']] # wins

# Instantiate a linear regression object
mlr = LinearRegression()

# Split the data into sections to train the model (80%) and test the model (20%)
x_train3, x_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=3)

# Fit a model to the training data
mlr_model = mlr.fit(x_train3, y_train3)

# Make some predictions based on the test x data
win_predict3 = mlr.predict(x_test3)

# Print out the results
print('Coefficients')
print(mlr.coef_)

print("Train score:")
print(mlr.score(x_train3, y_train3))

print("Test score:")
print(mlr.score(x_test3, y_test3))

fig3 = plt.figure()
plt.scatter(y_test3, win_predict3)
plt.show()