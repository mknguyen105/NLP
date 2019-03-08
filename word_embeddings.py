import sys, nltk, operator
from sklearn.metrics.pairwise import cosine_similarity
from word2vec_extractor import Word2vecExtractor
from dependency_demo_stub import  find_main
from qa_engine.base import QABase

def compare_words(word1, word2, W2vecextractor):
    word1_feat = W2vecextractor.word2v(word1)
    word2_feat = W2vecextractor.word2v(word2)
    dist = cosine_similarity([word1_feat], [word2_feat])
    dist = dist[0][0]
    return dist

