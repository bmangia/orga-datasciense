import pandas as pd
from matplotlib import pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv("../data/train.csv", header=0, sep=',',encoding='latin1')

print("Limpiando dataset...")
data_train_clean = data_train.apply(lambda row: clean_text(row['Text']), axis=1)
print("OK\n")

print("Creando bag of words...\n")
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 100) #probar con 5000 y 10000 con maquina mas potente

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(data_train_clean)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
print("OK\n")

print ("Entrenando el random forest...")
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 10) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, data_train["Prediction"] )
print ("OK\n")

print ("Calculando predicciones...")
data_test = pd.read_csv("../data/test.csv", header=0, sep=',',encoding='latin1')

data_test_clean = data_test.apply(lambda row: clean_text(row['Text']), axis=1)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(data_test_clean)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Id":data_test["Id"], "Prediction":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "../data/First_Bag_of_Words_and_Random_Forest_model.csv", index=False, quoting=3 )
print ("OK\n")