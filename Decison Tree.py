#DATA VISULAIZATION
import pandas
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/san_fran_crime.csv

#Import the data from the .csv file
dataset = pandas.read_csv('san_fran_crime.csv', delimiter="\t")

#Let's have a look at the data and the relationship we are going to model
print(dataset.head())
print(dataset.shape)

import graphing # custom graphing code. See our GitHub repo for details
import numpy as np

# Crime category
graphing.multiple_histogram(dataset, label_x='Category', label_group="Resolution", histfunc='sum', show=True)

# District
graphing.multiple_histogram(dataset, label_group="Resolution", label_x="PdDistrict", show=True)



# Map of crimes
graphing.scatter_2D(dataset, label_x="X", label_y="Y", label_colour="Resolution", title="GPS Coordinates", size_multiplier=0.8, show=True)

# Day of the week
graphing.multiple_histogram(dataset, label_group="Resolution", label_x="DayOfWeek", show=True)

# day of the year
# For graphing we simplify this to week or the graph becomes overwhelmed with bars
dataset["week_of_year"] = np.round(dataset.day_of_year / 7.0)
graphing.multiple_histogram(dataset, 
                    label_x='week_of_year',
                    label_group='Resolution',
                    histfunc='sum', show=True)
del dataset["week_of_year"]



#FINAL DATA REPRESENTATION
# One-hot encode categorical features
dataset = pandas.get_dummies(dataset, columns=["Category", "PdDistrict"], drop_first=False)



from sklearn.model_selection import train_test_split

# Split the dataset in an 90/10 train/test ratio. 
# We can afford to do this here because our dataset is very very large
# Normally we would choose a more even ratio
train, test = train_test_split(dataset, test_size=0.1, random_state=2, shuffle=True)

print("Data shape:")
print("train", train.shape)
print("test", test.shape)


#MODEL ASSESMENT CODE
from sklearn.metrics import balanced_accuracy_score

# Make a utility method that we can re-use throughout this exercise
# To easily fit and test out model

features = [c for c in dataset.columns if c != "Resolution"]


def fit_and_test_model(model):
    '''
    Trains a model and tests it against both train and test sets
    '''  
    global features

    # Train the model
    model.fit(train[features], train.Resolution)

    # Assess its performance
    # -- Train
    predictions = model.predict(train[features])
    train_accuracy = balanced_accuracy_score(train.Resolution, predictions)

    # -- Test
    predictions = model.predict(test[features])
    test_accuracy = balanced_accuracy_score(test.Resolution, predictions)

    return train_accuracy, test_accuracy


print("Ready to go!")



#FITTING A DECISON TRREE
import sklearn.tree

# fit a simple tree using only three levels
model = sklearn.tree.DecisionTreeClassifier(random_state=2, max_depth=3) 
train_accuracy, test_accuracy = fit_and_test_model(model)

print("Model trained!")
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)


#IMPROVING PERFORMANCE THROUGH ACHITECTURE
# fit a very deep tree
model = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=100)

train_accuracy, test_accuracy = fit_and_test_model(model)
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)


# Temporarily shrink the training set to something
# more realistic
full_training_set = train
train = train[:100]

# fit the same tree as before
model = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=100)

# Assess on the same test set as before
train_accuracy, test_accuracy = fit_and_test_model(model)
print("Train accuracy", train_accuracy)
print("Test accuracy", test_accuracy)

# Roll the training set back to the full set
train = full_training_set




print(dataset.head())
