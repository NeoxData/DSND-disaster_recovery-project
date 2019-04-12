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

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sqlalchemy as db


def load_data(database_filepath):
    engine = db.create_engine(database_filepath)
    conn= engine.connect()
    df = pd.read_sql_table('cat_messages', con=conn)

    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names=df.columns[4:]
    return X,Y,category_names


def tokenize(text):
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
    y_pred=model.predict(X_test)

    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    
    for i in range(len(category_names)):
        print('\n Category name: {} \n {} '.format( category_names[i], classification_report(Y_test[:,i].astype(int), y_pred[:,i].astype(int))))


def save_model(model, model_filepath):
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