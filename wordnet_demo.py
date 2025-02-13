import csv
from collections import defaultdict
from nltk.corpus import wordnet as wn
import qa


DATA_DIR = "./wordnet"

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

def find_hyponyms(word, graph):
    synsets = wn.synsets(word)
    for synset in synsets:
        hyponyms = synset.hyponyms()
        for hyponym in hyponyms:
            hyponym = hyponym.name()[0:hyponym.name().index(".")]
            #print("Hyponym:" + str(hyponym))
            node = qa.find_node(hyponym, graph)
            if node is not None:
                return node

def find_hypernyms(word, graph):
    synsets = wn.synsets(word)
    for synset in synsets:
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            hypernym = hypernym.name()[0:hypernym.name().index(".")]
            #print("Hypernym:" + str(hypernym))
            node = qa.find_node(hypernym, graph)
            if node is not None:
                return node
        return None

def word_in_string(word, q_graph, q_text):
    hypernym = find_hypernyms(word, q_graph)
    hyponym = find_hyponyms(word, q_graph)

    if hypernym is not None:
        if hypernym['word'] in q_text:
            return True

    if hyponym is not None:
        if hyponym['word'] in q_text:
            return True

    if word in q_text:
        return True

    return False


if __name__ == "__main__":

    ## You can use either the .csv files or the .dict files.
    ## If you use the .dict files, you MUST use "rb"!

    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))


    # {synset_id : {synset_offset: X, noun/verb: Y, stories: set(Z)}}, ...}
    # e.g. {help.v.01: {synset_offset: 2547586, noun: aid, stories: set(Z)}}, ...
    #noun_ids = pickle.load(open("Wordnet_nouns.dict", "rb"))
    #verb_ids = pickle.load(open("Wordnet_verbs.dict", "rb"))

    nouns = []
    # iterate through dictionary
    for synset_id, items in noun_ids.items():
        noun = items['story_noun']
        stories = items['stories']
        nouns.append(noun)
        # print(noun, stories)
        # get lemmas, hyponyms, hypernyms

    for synset_id, items in verb_ids.items():
        verb = items['story_verb']
        stories = items['stories']
        # print(verb, stories)
        # get lemmas, hyponyms, hypernyms


    # 'Rodent' is a hypernym of 'mouse',
    # so we look at hyponyms of 'rodent' to find 'mouse'
    #
    # Question: Where did the rodent run into?
    # Answer: the face of the lion
    # Sch: The lion awaked because a mouse ran into the face of the lion.
    rodent_synsets = wn.synsets("rodent")
    print("'Rodent' synsets: %s" % rodent_synsets)

    print("'Rodent' hyponyms")
    for rodent_synset in rodent_synsets:
        rodent_hypo = rodent_synset.hyponyms()
        print("%s: %s" % (rodent_synset, rodent_hypo))

        for hypo in rodent_hypo:
            print(hypo.name()[0:hypo.name().index(".")])
            print("is hypo_synset in Wordnet_nouns/verbs.csv?")
            # match on "mouse.n.01"
            if hypo.name()[0:hypo.name().index(".")] in nouns:
                print('yes')


    # 'Know' is a hyponym of 'recognize' (know.v.09),
    # so we look at hypernyms of 'know' to find 'recognize'
    #
    # Question: What did the mouse know?
    # Answer: the voice of the lion
    # Sch: The mouse recognized the voice of the lion.
    know_synsets = wn.synsets("try")
    print("\n'Know' synsets: %s" % know_synsets)

    print("'Know' hypernyms")
    for know_synset in know_synsets:
        print(know_synset.lemma_names())
        print(know_synset.name()[0:know_synset.name().index(".")])
        know_hyper = know_synset.hypernyms()
        for hyper in know_hyper:
            syn = wn.synsets(hyper.name()[0:hyper.name().index(".")])
            print('syn: ', syn)
        print("%s: %s" % (know_synset, know_hyper))

    # 'Express mirth' is a lemma of 'laugh'
    # so we look at lemmas of 'express mirth' to find 'laugh'
    #
    # Question: Who expressed mirth?
    # Answer: the lion
    # Sch: The lion laughed aloud because he thought that the mouse is extremely not able to help him.
    mirth_synsets = wn.synsets("express_mirth")
    print("\n'Express Mirth' synsets: %s" % mirth_synsets)

    print("'Express mirth' lemmas")
    for mirth_synset in mirth_synsets:
        print(mirth_synset)

        # look up in dictionary
        print("\n'%s' is in our dictionary: %s" % (mirth_synset.name(), (mirth_synset.name() in verb_ids)))


