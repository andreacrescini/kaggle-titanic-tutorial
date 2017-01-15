# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pandas.read_csv("titanic_train.csv")

##2: Looking At The Data
#Describe the dataset using print(titanic.describe()).
# Print the first 5 rows of the dataframe.
print(titanic.head(5))
print(titanic.describe())

##3: Missing Data
#Replace all the missing values in the Age column of titanic.
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

##4 Non-Numeric Columns
#We have to either exclude our non-numeric columns when we train our algorithm 
#Name, Sex, Cabin, Embarked, and Ticket, or find a way to convert them to numeric columns.
#We'll ignore the Ticket, Cabin, and Name columns. Too many missing values 

##5: Converting The Sex Column
# Find all the unique genders -- the column appears to contain only male and female.
print(titanic["Sex"].unique())
# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#6: Converting The Embarked Column
# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())
titanic.loc[titanic["Embarked"].isnull() , "Embarked"] = "S"
titanic.loc[titanic["Embarked"] == "S" , "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C" , "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q" , "Embarked"] = 2 

##7: On To Machine Learning!

##8: Cross Validation
#We can now use linear regression to make predictions on our training set.
#We want to train the algorithm on different data than we make predictions on. 
#This is critical if we want to avoid overfitting. 
#Overfitting is what happens when a model fits itself to "noise", not signal.

#Every machine learning algorithm can overfit, although some (like linear regression) 
#are much less prone to it. If you evaluate your algorithm on the same dataset that you
#train it on, it's impossible to know if it's performing well because it overfit 
#itself to the noise, or if it actually is a good algorithm.

#Luckily, cross validation is a simple way to avoid overfitting. To cross validate, 
#you split your data into some number of parts (or "folds"). Lets use 3 as an example. 
#You then do this:
#   Combine the first two parts, train a model, make predictions on the third.
#   Combine the first and third parts, train a model, make predictions on the second.
#   Combine the second and third parts, train a model, make predictions on the first.

#9: Making Predictions
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
#shape[0] is the number of rows
kf = KFold(titanic.shape[0], n_folds=3, random_state=1) ## this is equivalente to: sklearn.cross_validation.KFold(n= titanic.shape[0], n_folds=3, shuffle=False,random_state=1) 

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
##10: Evaluating Error
#We'll first need to define an error metric, so we can figure out how accurate our model is.
#From the Kaggle competition description, the error metric is percentage of correct 
#predictions. We'll use this same metric to evaluate our performance locally.
#The metric will basically involve finding the number of values in predictions that are the 
#exact same as their counterparts in titanic["Survived"], and then dividing by the total number of passengers.
#Before we can do this, we need to combine the 3 sets of predictions into one column. 
#Since each set of predictions is a numpy (python scientific computing library) array, 
#we can use a numpy function to concatenate them into one.

import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

##11: Logistic Regression
#We have our first predictions! They aren't very good, though, with only 78.3% accuracy. 
#We can instead use logistic regression to instead output values between 0 and 1.

#One good way to think of logistic regression is that it takes the output of a linear 
#regression, and maps it to a probability value between 0 and 1. The mapping is done using 
#the logit function. Passing any value through the logit function will map it to a value 
#between 0 and 1 by "squeezing" the extreme values. This is perfect for us, because we only 
#care about two outcomes.

#Sklearn has a class for logistic regression that we can use. We'll also make things easier 
#by using an sklearn helper function to do all of our cross validation and evaluation for us.
from sklearn import cross_validation

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

##12: Processing The Test Set
#We need to make a submission to the competition. To do this, we need to take the exact 
#same steps on the test data that we took on the training data. If we don't do the exact 
#same operations, then we won't be able to make valid predictions on it.

#These operations are all the changes we made to the columns before.
#Process titanic_test the same way we processed titanic:
#- Replace the missing values in the "Age" column with the median age from the train set. 
#The age has to be the exact same value we replaced the missing ages in the training set 
#with (IT CANNOT BE THE MEDIA OF THE TEST SET, BECAUSE THIS IS DIFFERENT ). 
#You should use titanic["Age"].median() to find the median.
#   X Replace any male values in the Sex column with 0, and any female values with 1.
#   X Fill any missing values in the Embarked column with S.
#   X In the Embarked column, replace S with 0, C with 1, and Q with 2.

#We'll also need to replace a missing value in the Fare column.
#   X Use .fillna with the median of the column in the test set to replace this.
#   X There are no missing values in the Fare column of the training set, but test sets 
#     can sometimes be different.
titanic_test = pandas.read_csv("titanic_test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

##13: Generating A Submission File
#Now we have everything we need to generate a submission for the competition!

#First, we have to train an algorithm on the training data. 
#Then, we make predictions on the test set. 
#Finally, we'll generate a csv file with the predictions and passenger ids.

#Once you finish all the steps in the code below, you can output to a csv by using 
#submission.to_csv("kaggle.csv", index=False).

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    

#################################
#1: Improving Our Features
#Our submission wasn't very high-scoring, though. There are three main ways we can improve it:
#   X Use a better machine learning algorithm.
#   X Generate better features.
#   X Combine multiple machine learning algorithms.

#2: Random Forest Introduction
#3: Implementing A Random Forest
#Make cross validated predictions for the titanic dataframe (which has already been loaded 
#in) using 3 folds.
#   X Use the random forest algorithm stored in alg to do the cross validation.
#   X Use predictors to predict the Survived column. Assign the result to scores.
#   X You can use the cross_validation.cross_val_score function to do this.
#       o You'll need to initialize an instance of KFold like we did in the last mission, and pass it into the cv keyword argument of the cross_val_score function.
#After making cross validated predictions, print out the mean of scores.

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

##4: Parameter Tuning
#The first, and easiest, thing we can do to improve the accuracy of the random forest is to
#increase the number of trees we're using. Training more trees will take more time, 
#but because of the fact that we're averaging many predictions made on different subsets 
#of the data, having more trees will increase accuracy greatly (up to a point).

#We can also tweak the min_samples_split and min_samples_leaf variables to reduce 
#overfitting. Because of how a decision tree works, having splits that go all the way down,
#or overly deep in the tree can result in fitting to quirks in the dataset, and not true 
#signal. Thus, increasing min_samples_split and min_samples_leaf can reduce overfitting, 
#which will actually improve our score, as we're making predictions on unseen data. 
#A model that is less overfit, and that can generalize better, will actually perform better 
#on unseen data, but worse on seen data.

#We've changed the parameters used when we initialize alg. We'll need to re-run our model now:
#   X Make cross validated predictions for the titanic dataframe using 3 folds.
#   X Use predictors to predict the Survived column and assign the result to scores.
#   X After making cross validated predictions, print out the mean of scores.

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

##5: Generating New Features
#An easy way to generate features is to use the .apply method on pandas dataframes. 
#This applies a function you pass in to each element in a dataframe or series. 
#We can pass in a lambda function, which enables us to define a function inline.

#To write a lambda function, you write lambda x: len(x). x will take on the value of the 
#input that is passed in -- in this case, the passenger name. The function to the right of 
#the colon is then applied to x, and the result returned. The .apply method takes all 
#of these outputs and constructs a pandas series from them. We can assign this series to a
# dataframe column.

# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

##6: Using The Title
#We can extract the title of the passenger from their name. 
#The titles take the form of Master., Mr., Mrs.. There are a few very commonly used titles, 
#and a "long tail" of one-off titles that only one or two passengers have.

#We'll first extract the titles with a regular expression, and then map each unique title 
#to an integer value.

import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# Verify that we converted everything.
print(pandas.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles

##7: Family Groups
#We can also generate a feature indicating which family people are in. 
#Because survival was likely highly dependent on your family and the people around you, 
#this has a good chance at being a good feature.

#To get this, we'll concatenate someone's last name with FamilySize to get a unique family 
#id. We'll then be able to assign a code to each person based on their family id.
import operator

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1) # argument of apply: 0 or ‘index’: apply function to each column; 1 or ‘columns’: apply function to each row.

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[titanic["FamilySize"] < 3] = -1

# Print the count of each unique id.
print(pandas.value_counts(family_ids))

titanic["FamilyId"] = family_ids

##8: Finding The Best Features
#Feature engineering is the most important part of any machine learning task, and there are
#lots more features we could calculate. But we also need a way to figure out which features
#are the best.

#One way to do this is to use univariate feature selection. This essentially goes column by
#column, and figures out which columns correlate most closely with what we're trying to 
#predict (Survived).

#As usual, sklearn has a function that will help us with feature selection, SelectKBest. 
#This selects the best features from the data, and allows us to specify how many it selects.

#We've updated predictors. Make cross validated predictions for the titanic dataframe using 3 folds.
#Use predictors to predict the Survived column and assign the result to scores.
#After making cross validated predictions, print out the mean of scores.
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

##9: Gradient Boosting
#Another method that builds on decision trees is a gradient boosting classifier. 
#Boosting involves training decision trees one after another, and feeding the errors from one
#tree into the next tree. So each tree is building on all the other trees that came before
#it. This can lead to overfitting if we build too many trees, though. As you get above 100 
#trees or so, it's very easy to overfit and train to quirks in the dataset. As our dataset 
#is extremely small, we'll limit the tree count to just 25.

#Another way to limit overfitting is to limit the depth to which each tree in the gradient 
#boosting process can be built. We'll limit the tree depth to 3 to avoid overfitting.

#We'll try boosting instead of our random forest approach and see if we can improve our accuracy.

#10: Ensembling
#One thing we can do to improve the accuracy of our predictions is to ensemble different 
#classifiers. Ensembling means that we generate predictions using information from a set of 
#classifiers, instead of just one. In practice, this means that we average their predictions.

#Generally, the more diverse the models we ensemble, the higher our accuracy will be. 
#Diversity means that the models generate their results from different columns, or use a 
#very different method to generate predictions. Ensembling a random forest classifier with
#a decision tree probably won't work extremely well, because they are very similar. 
#On the other hand, ensembling a linear regression with a random forest can work very well.
 
#One caveat with ensembling is that the classifiers we use have to be about the same in 
#terms of accuracy. Ensembling one classifier that is much worse than another probably will
#make the final result worse.

#In this case, we'll ensemble logistic regression trained on the most linear predictors 
#(the ones that have a linear ordering, and some correlation to Survived), and a gradient
#boosted tree trained on all of the predictors.
                                                                                      #
#We'll keep things simple when we ensemble -- we'll average the raw probabilities (from 0 
#to 1) that we get from our classifiers, and then assume that anything above .5 maps to one,
#and anything below or equal to .5 maps to 0.

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)

#11: Matching Our Changes On The Test Set
#There are a lot of things we could do to make this analysis better that we'll talk about at
#the end, but for now, let's make a submission.

#The first step is matching all our training set changes on the test set data, like we did 
#in the last mission. We've read the test set into titanic_test. We'll have to match our changes:

#   X Generate the NameLength column, which is how long the name is.
#   X Generate the FamilySize column, showing how large a family is.
#   X Add in the Title column, keeping the same mapping that we had before.
#   X Add in a FamilyId column, keeping the ids consistent across the train and test sets.
#Add the NameLength column to titanic_test.
#Do this the same way we did it with the titanic dataframe.

# First, we'll add titles to the test set.
titles = titanic_test["Name"].apply(get_title)
# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
# Check the counts of each unique title.
print(pandas.value_counts(titanic_test["Title"]))

# Now, we add the family size column.
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# Now we can add family ids.
# We'll use the same ids that we did earlier.
print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#12: Predicting On The Test Set
#We have some better predictions now, so let's create another submission.
#   X Turn the predictions into either 0 or 1 by turning the predictions less than or equal to .5 into 0, and the predictions greater than .5 into 1.
#   X Then, convert the predictions to integers using the .astype(int) method -- if you don't, Kaggle will give you a score of 0.
#   X Finally, create a submission dataframe where the first column is PassengerId, and the second column is Survived (this will be the predictions).
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    
##14: Final Thoughts    
#Now, we have a submission! It should get you a score of .799 on the leaderboard. 
#You can generate a submission file with submission.to_csv("kaggle.csv", index=False).

#There's still more work you can do in feature engineering:
#   X Try using features related to the cabins.
#   X See if any family size features might help -- do the number of women in a family make the whole family more likely to survive?
#   X Does the national origin of the passenger's name have anything to do with survival?

#There's also a lot more we can do on the algorithm side:
#   X Try the random forest classifier in the ensemble.
#   X A support vector machine might work well with this data.
#   X We could try neural networks.
#   X Boosting with a different base classifier might work better.

#And with ensembling methods:
#   X Could majority voting be a better ensembling method than averaging probabilities?

#This dataset is very easy to overfit on because there isn't a lot of data, so you'll be grinding for small accuracy gains.
