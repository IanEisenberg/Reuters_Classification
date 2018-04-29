import nltk
from nltk import word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# taken from https://gist.github.com/miguelmalvarez/31122eb9e4c0af8adeca
cachedStopWords = stopwords.words("english")
lemma = nltk.wordnet.WordNetLemmatizer()

# strips text down
def tokenize(text, lemmatize=True, vocab=None):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    if lemmatize:
        words =(list(map(lambda token: lemma.lemmatize(token), words)));
    if vocab:
        words = [word for word in words if vocab.get(word) is not None]
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token:
                          p.match(token) and len(token)>=min_length, words));
    return filtered_tokens

# not building a recurrent network, so we need a way to weight the different words.
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                        max_df=0.90, max_features=30000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');
    tfidf.fit(docs);
    return tfidf;

def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index])
                 for index in doc_representation.nonzero()[1]]