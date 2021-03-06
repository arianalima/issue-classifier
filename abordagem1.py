import sys

import numpy as np
import pandas as pd
import re

import spacy
from scipy.stats import expon, uniform
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import bot.BotNotify as Bot


def lemmatizer(filtered_tokens):
    lemma = []
    for sentence in filtered_tokens:
        lemma_part = []
        for doc in sentence:
            for token in doc:
                lemma_part.append(token.lemma_.lower())
        lemma.append(lemma_part)
    return lemma


def remove_stop_words(text_list):
    nlp = spacy.load('en_core_web_md') #en_core_web_sm
    token_list = []
    for text in text_list:
        symbols = set(re.findall("[^\w\s]", text))
        urls = set(re.findall("(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\."
                              "[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]"
                              "+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", text))
        for url in urls:
            text = text.replace(url, ' ')
        for sym in symbols:
            text = text.replace(sym, ' ')
        token_list.append(text.strip().split())

    filtered_tokens = []
    for sentence in token_list:
        filtered_tokens_part = []
        for word in sentence:
            lexeme = nlp.vocab[word]
            if not lexeme.is_stop:
                filtered_tokens_part.append(nlp(word))
        filtered_tokens.append(filtered_tokens_part)
    return filtered_tokens


def identify_tags(text, tag):
    text = text.replace('\r\n', ' ')
    snippet = re.findall("({%s.*?{%s})" % (tag, tag), text)
    for snip in snippet:
        text = text.replace(snip, 'tag_%s' % tag)
    return text


def count_vectorize(lemmatized_tokens):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(lemmatized_tokens)
    names = vectorizer.get_feature_names()
    return X.toarray()


def read_dataset(path):
    data = pd.read_csv(path, sep=';', encoding='utf-8')
    descriptions = data['Description']
    points = data['Custom field (Story Points)']

    TAGS = ['code', 'quote', 'html', 'noformat', 'panel']

    """Identifies and replaces the snippets for its tag"""
    for tag in TAGS:
        data['Description'] = list(map(lambda text: identify_tags(text, tag), descriptions))

    filtered_tokens = remove_stop_words(descriptions)
    lemmatized_tokens = lemmatizer(filtered_tokens)
    for index in range(len(lemmatized_tokens)):
        lemmatized_tokens[index] = " ".join(lemmatized_tokens[index])

    descriptions = count_vectorize(lemmatized_tokens)

    return descriptions, points


def classification_results(classifier, X, y, grid_result):

    # """ TRAINNING SCORE """
    # """ GETTING MEANS GRID RESULT ATTRIBUTES"""
    means_acc = grid_result.cv_results_['mean_test_accuracy']
    stds_acc = grid_result.cv_results_['std_test_accuracy']
    means_pre = grid_result.cv_results_['mean_test_precision_weighted']
    stds_pre = grid_result.cv_results_['std_test_precision_weighted']
    means_rec = grid_result.cv_results_['mean_test_recall_weighted']
    stds_rec = grid_result.cv_results_['std_test_recall_weighted']
    means_f1 = grid_result.cv_results_['mean_test_f1_weighted']
    stds_f1 = grid_result.cv_results_['std_test_f1_weighted']
    params = grid_result.cv_results_['params']

    # """ GETTING BEST INDEX FROM MEANS GRID RESULT ATTRIBUTES"""
    mean_acc = means_acc[grid_result.best_index_]
    stdev_acc = stds_acc[grid_result.best_index_]
    mean_pre = means_pre[grid_result.best_index_]
    stdev_pre = stds_pre[grid_result.best_index_]
    mean_rec = means_rec[grid_result.best_index_]
    stdev_rec = stds_rec[grid_result.best_index_]
    mean_f1 = means_f1[grid_result.best_index_]
    stdev_f1 = stds_f1[grid_result.best_index_]
    param = params[grid_result.best_index_]

    # """ GETTING TEST VALUES FROM MODEL """
    y_pred = grid_result.predict(X)  #X.docvecs.vectors_docs
    trainning_metrics = precision_recall_fscore_support(y, y_pred, average='micro')

    # """ GETTING CONFUSION MATRIX"""
    confusion_matrix = metrics.confusion_matrix(y, y_pred)

    """ summarize results """
    print("----------------------------------------------------------")
    print("Classifier: %s " % (classifier))
    print("----------------------------------------------------------")
    print("- Best Trainnig Score=%f " % (grid_result.best_score_))
    print("- Best Trainnig accuracy=%f (std=%f)" % (mean_acc, stdev_acc))
    print("- Best Trainnig precision=%f (std=%f)" % (mean_pre, stdev_pre))
    print("- Best Trainnig recall=%f (std=%f)" % (mean_rec, stdev_rec))
    print("- Best Trainnig f1=%f (std=%f)" % (mean_f1, stdev_f1))
    print("- Best Trainnig Param=%r" % (param))
    print("----------------------------------------------------------")
    print("- Test input classes: {}".format(np.unique(y)))
    print("- Test predicted classes: {}".format(np.unique(y_pred)))
    print("----------------------------------------------------------")
    print("- Test accuracy = %f " % (accuracy_score(y, y_pred)))
    print("- Test precision = %f " % (trainning_metrics[0]))
    print("- Test recall = %f " % (trainning_metrics[1]))
    print("- Test f1 = %f " % (trainning_metrics[2]))
    print("----------------------------------------------------------")
    print("Confusion Matrix")
    print("----------------------------------------------------------")
    print("```")
    print(confusion_matrix)
    print("```")
    print("----------------------------------------------------------")
    print("")
    print("")


def run_classification(X, y):
    pipeline = SVC()
    param_dist = {
        'C': expon(scale=1000, loc=1e-10),
        'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
        'degree': np.arange(1, 5),
        'gamma': uniform(loc=1e-10, scale=1)
    }
    n_iter_search = 100

    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    grid_result = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist,
                                     n_iter=n_iter_search, n_jobs=2, scoring=scoring,
                                     refit='f1_weighted', cv=10, verbose=0)

    grid_result.fit(X, y)
    classification_results("SVM", X, y, grid_result)


file = __file__.split("/")[-1]
Bot.notify("Início %s" % file)
np.random.seed(3)

descriptions, points = read_dataset(sys.argv[3])
run_classification(descriptions, points)

Bot.notify("Fim %s" % file)
