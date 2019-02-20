
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

import nltk
STOPWORDS = nltk.corpus.stopwords.words('english')

# Our simple grammar from class (and the book)
import nltk, re

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at", "behind", "below", "beside", "above", "across", "along", "below", "between", "under",
              "near", "inside"])

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def get_best_sentences(question, story):
    """
       Answer Identification
       - Basic Word Overlap:
       - Stop Words:
       - Roots:
       - Weights: (verbs might be given more weight than nouns)
       - Similar Words

       Pipeline:
       Get subj and verb from question
       Get Question Type
       Search through sentences and return sentences that have the subject

       """

    best_sentences = {}
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    story_subjects = get_story_subjects(story['dep'])

    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])


    question_text = question['text']
    question_words = nltk.word_tokenize(str(question_text))

    type = get_question_type(question)

    print(question_text)

    #Classify question type
    verb_count = 0
    noun_count = 0
    keywords = []
    if type == "what":
        verb_count = 3
        noun_count = 2
    elif type == "where":
        #Look for prepositional noun
        keywords = ["in", "on", "at", "behind", "below", "beside", "above", "across", "along", "below", "between", "under",
              "near", "inside"]
        key_count = 1
        verb_count = 2
        noun_count = 2
    elif type == "who":
        #Look for person
        verb_count = 2
        noun_count = 3
    elif type == "why":
        key_count = 1
        keywords = ['because', 'so that', 'in order to']


    for sent in sentences:
        count = 0
        for token, pos in sent:
            if token.lower() in keywords:
                count = count + key_count
            for qword in question_words:
                if pos == "VBP" or pos == "VB":
                   # count += get_verb_similarity(qword, token)
                    pass
                if stemmer.stem(qword) == stemmer.stem(token) and token not in STOPWORDS:
                #if stemmer.stem(qword) == stemmer.stem(token):
                    if pos == "VBP" or pos == "VB":
                        count = count + verb_count
                    elif pos == "NN" or pos == "NNP" or pos == "NNS":
                        count = count + noun_count
                    else:
                        count = count + 1

        tokenized_sentence = ' '.join([word for (word,tag) in sent])
        print(tokenized_sentence + " " + str(count))
        best_sentences[tokenized_sentence] = count

    return best_sentences

def get_sentence(sentences):
    """
    Loops through dictionary (sentence: count of words that match) and chooses the sentence with the highest count
    """
    best_number = 0
    best_sentence = ""
    for s in sentences:
        if(sentences[s] > best_number):
            best_number = sentences[s]
            best_sentence = s
    return best_sentence

def get_verb(dep):
    i = 0
    verb = None
    dep_graph = dep['dep']
    while dep_graph.contains_address(i):
        if dep_graph._rel(i) == 'root':
            verb = dep_graph.get_by_address(i)['word']
        i += 1
    lmtzr = nltk.WordNetLemmatizer()
    verb = lmtzr.lemmatize(verb, 'v')
    return verb


def get_nsubj(dep):
    nsubjs = []
    i = 0
    dep_graph = dep['dep']
    while dep_graph.contains_address(i):
        if dep_graph._rel(i) == 'nsubj':
            word = dep_graph.get_by_address(i)['word']
            nsubjs.append(word)
        if dep_graph._rel(i) == 'nmod':
            word = dep_graph.get_by_address(i)['word']
            nsubjs.append(word)
        i += 1

    return nsubjs



def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None:
        return root

    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:
        return root

    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild)
            if match is None:
                return None
        return root

    return None

# This returns who what when where why or how
# I know for this assignment we could probably just use the first word and exclude the funtion, but I figure this will
# keep us organized if we're later thrown questions that don't have the question word as the first word, or if there is
# no question word in the question at all
def get_question_type(question):
    question_types = ['who', 'what', 'when', 'where', 'why']

    words = nltk.word_tokenize(question['text'])
    first_word = words[0].lower()
    if first_word in question_types:
        return first_word

# Add other methods of detecting question type to cover it not being first word
    # Who

    # What

    # When

    # Where

    # Why

    # How


def get_locations(best_sentences, chunker):
    print("Get locations")

def get_verb_similarity(verb1, verb2):
    score = 0
    try:
        word1 = wn.synsets(verb1, pos='v')
        word2 = wn.synsets(verb2, pos='v')
        for syn1 in word1:
            for syn2 in word2:
                if syn1.path_similarity(syn2) == 1:
                    return 1
                elif syn1.path_similarity(syn2) > 5:
                    score = 0
    except:
        return 0

    return score

def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    ###     Your Code Goes Here         ###
    """

    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
    best_sentences = get_best_sentences(question, story)
    answer = get_sentence(best_sentences)

    return answer


    ###     End of Your Code         ###





#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

#############################################################

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
