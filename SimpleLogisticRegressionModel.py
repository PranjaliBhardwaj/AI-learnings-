# DATA VISUALIZATION
import pandas
!pip install statsmodels
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/avalanche.csv

#Import the data from the .csv file
dataset = pandas.read_csv('avalanche.csv', delimiter="\t")

#Let's have a look at the data
dataset


# DATA EXPLORATION
import graphing # custom graphing code. See our GitHub repo for details

graphing.box_and_whisker(dataset, label_x="avalanche", label_y="surface_hoar", show=True)
graphing.box_and_whisker(dataset, label_x="avalanche", label_y="fresh_thickness", show=True)
graphing.box_and_whisker(dataset, label_x="avalanche", label_y="weak_layers", show=True)
graphing.box_and_whisker(dataset, label_x="avalanche", label_y="no_visitors")



#BUILDING A SIMPLE LOGISTIC REGRESSION MODEL
# Here we import a function that splits datasets according to a given ratio
from sklearn.model_selection import train_test_split

# Split the dataset in an 70/30 train/test ratio. 
train, test = train_test_split(dataset, test_size=0.3, random_state=2)
print(train.shape)
print(test.shape)


#TRAIN DATASET

import statsmodels.formula.api as smf
import graphing # custom graphing code. See our GitHub repo for details

# Perform logistic regression.
model = smf.logit("avalanche ~ weak_layers", train).fit()

print("Model trained")


print(model.summary())

 #USING OUR MODEL
# predict to get a probability

# get first 3 samples from dataset
samples = test["weak_layers"][:4]

# use the model to get predictions as possibilities
estimated_probabilities = model.predict(samples)

# Print results for each sample
for sample, pred in zip(samples,estimated_probabilities):
    print(f"A weak_layer with value {sample} yields a {pred * 100:.2f}% chance of an avalanche.")



# plot the model
# Show a graph of the result
predict = lambda x: model.predict(pandas.DataFrame({"weak_layers": x}))

graphing.line_2D([("Model", predict)],
                 x_range=[-20,40],
                 label_x="weak_layers", 
                 label_y="estimated probability of an avalanche")

print("Minimum number of weak layers:", min(train.weak_layers))
print("Maximum number of weak layers:", max(train.weak_layers))


#WHEN THERE IS 0 LAYERS THERE IS SOME CHANCES OF AVLANCHE , SO FROM 10 LAYERS THERE WON'T BE CHANCES OF AVLANCHE. THAT WAS IN LINE
import numpy as np

# Get actual rates of avalanches at 0 layers
avalanche_outcomes_for_0_layers = train[train.weak_layers == 0].avalanche
print("Average rate of avalanches for 0 weak layers of snow", np.average(avalanche_outcomes_for_0_layers))

# Get actual rates of avalanches at 10 layers
avalanche_outcomes_for_10_layers = train[train.weak_layers == 10].avalanche
print("Average rate of avalanches for 10 weak layers of snow", np.average(avalanche_outcomes_for_10_layers))

#CLASSIFICATION OR DECISON THRESOLD
# threshold to get an absolute value
threshold = 0.5

# Add classification to the samples we used before
for sample, pred in list(zip(samples,estimated_probabilities)):
    print(f"A weak_layer with value {sample} yields a chance of {pred * 100:.2f}% of an avalanche. Classification = {pred > threshold}")

#PERFORM ON TEST SET
# Classify the mdel predictions using the threshold
predictions = model.predict(test) > threshold

# Compare the predictions to the actual outcomes in the dataset
accuracy = np.average(predictions == test.avalanche)

# Print the evaluation
print(f"The model correctly predicted outcomes {accuracy * 100:.2f}% of time.")





