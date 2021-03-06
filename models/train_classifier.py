import sys
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sqlalchemy as db


def load_data(database_filepath):
    '''
    Load data from database as dataframe

    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    '''
    engine = db.create_engine('sqlite:///' + database_filepath)
    conn= engine.connect()
    df = pd.read_sql_table('cat_messages', con=conn)

    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names=df.columns[4:]
    return X,Y,category_names


def tokenize(text):
    '''
    Tokenize and clean text

    Input:
        text: original message text
    Output:
        clean_tokens: Tokenized, stop words removed, and lemmatized text
    '''
     # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    st_words = [w for w in words if w not in stopwords.words("english")]
    
    #lemmatizing
    lemmatizer=WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in st_words:
        clean_tok=lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Build a ML pipeline using countvectorizer,ifidf, random forest classifier and gridsearch

    Input: None
    Output:
        Results of GridSearchCV
    '''
    pipeline = Pipeline([
        ('text_pipeline', Pipeline ([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('multi', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {'text_pipeline__vect__max_df': (0.4,0.5),
                 'multi__estimator__n_estimators': [10, 15]
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data

    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True labels for test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''

    y_pred=model.predict(X_test)
    
    #Getting accuracy of the model
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)

    #Getting for each category some performance indicators(precision, recall,f1-score,support)
    for i in range(len(category_names)):
        print('\n Category name: {} \n {} '.format( category_names[i], classification_report(Y_test[:,i].astype(int), y_pred[:,i].astype(int))))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 

    Input: 
        model: Model to be saved
        model_filepath: Path of the output pickle file
    Output:
        A pickle file of saved model
    '''

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