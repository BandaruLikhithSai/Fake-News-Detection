import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# helper for building the vectorizer (does not remove the word 'not')
def build_vectorizer():
    from sklearn.feature_extraction import text as sktext
    stop_words = set(sktext.ENGLISH_STOP_WORDS) - {'not'}
    return TfidfVectorizer(stop_words=list(stop_words), ngram_range=(1,2), max_df=0.85)


def train(save=True):
    """Train the model and optionally save artifacts."""
    print('Loading dataset...')
    data = pd.read_csv('news.csv')
    data['label'] = data['label'].str.upper()
    X = data['text']; y = data['label']

    tfidf = build_vectorizer()
    X_tfidf = tfidf.fit_transform(X)
    print('TF-IDF completed')

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)
    print('Model trained successfully')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percent = round(accuracy * 100, 2)
    print('Model Accuracy:', accuracy_percent, '%')

    if save:
        pickle.dump(model, open('model.pkl','wb'))
        pickle.dump(tfidf, open('tfidf.pkl','wb'))
        pickle.dump(accuracy_percent, open('accuracy.pkl','wb'))
        print('Saved model + tfidf + accuracy')

    return model, tfidf, accuracy_percent


if __name__ == '__main__':
    train(save=True)
