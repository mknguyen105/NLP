import sys, nltk, operator
from sklearn.metrics.pairwise import cosine_similarity
from word2vec_extractor import Word2vecExtractor
from dependency_demo_stub import  find_main
from qa_engine.base import QABase
import dependency_stub
import nltk.corpus

def compare_words(word1, word2, W2vecextractor):
    word1_feat = W2vecextractor.word2v(word1)
    word2_feat = W2vecextractor.word2v(word2)
    dist = cosine_similarity([word1_feat], [word2_feat])
    dist = dist[0][0]
    return dist

def get_similarity_score(word1, word2, W2vecextractor, modifier):
    score = 0
    h_nyms = dependency_stub.find_all_h_nyms([word1])
    if word2 in h_nyms and word1 != 'be':
        score += 1
        print('Matched', word2, 'to', word1)
    score += compare_words(word1, word2, W2vecextractor)
    if score > 0:
        score += modifier
    return score