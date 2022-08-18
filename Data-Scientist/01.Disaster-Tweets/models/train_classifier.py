import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import regex as re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pickle


def load_data(database_filepath):
    """
    Load data

    Arguments:
        database_filepath(db): path to the database file

    Outputs:
        X(df): input data for training
        Y(df): output data for training
        category_names: output labels
    """
    engine = create_engine('sqlite:///../data/'+database_filepath)
    df = pd.read_sql_table('DisasterData',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X, y, category_names


def tokenize(text):
    """
    Text preprocessing
    
    Arguments:
        text(str): raw text
        
    Outputs:
        clean_tokens(list): tokenized processed text
    """
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower()) # remove punctuation, lowercase
    words = word_tokenize(text) # tokenize
    words = [ w for w in words if w not in stopwords.words("english") ] # remove stopwords
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [ lemmatizer.lemmatize(w).strip() for w in words ] # lemmatize words, remove extra spaces
        
    return clean_tokens


def build_model():
    """
    Build model pipeline
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(
                tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(
                RandomForestClassifier())),
        ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model predictions
    
    Arguments:
        y_test: true classes
        preds: predicted classes
        
    Outputs
        classification_report: classification report for every label
        f1: average F1 score across labels
        accuracy: average accuracy across labels
    """
    preds = model.predict(X_test)
    for i,col in enumerate(category_names):
        # if i in [0, 9]: continue  # Skip bad column, TODO: Fix 0th column to not be 3 classes for no reason
        print(col)

        f1 = []
        accuracy = []
        y_true = list(y_test.values[:, i])
        y_pred = list(preds[:, i])
        f1.append(f1_score(y_true,y_pred))
        accuracy.append(accuracy_score(y_true,y_pred))
        print(classification_report(y_true, y_pred))

    print("Average F1 score:", np.mean(f1), "\nAverage Accuracy:", np.mean(accuracy))


def save_model(model, model_filepath):
    """
    Save model in a pickle file

    Arguments:
        model: model to save
        model_filepath: path to the model to save
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()