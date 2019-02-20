from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk, re, operator
import baseline
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


STOPWORDS = set(nltk.corpus.stopwords.words("english"))

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


'''
def get_best_sentences(patterns, sentences):
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                matches = False
            else:
                matches = True
        if matches:
            result.append(sent)

        return result
'''

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

    best_sentences = []
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")

    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])


    question_text = question['text']
    question_words = nltk.word_tokenize(str(question_text))

    q_type = get_question_type(question)

    nsubj = get_nsubj(question)
    verb = get_verb(question)

    print(question_text)
    print(nsubj)
    print(verb)

    #Classify question type
    verb_count = 0
    noun_count = 0

    if q_type == "what":
        verb_count = 2
        noun_count = 2
    elif q_type == "where":
        #Look for prepositional noun
        verb_count = 2
        noun_count = 3
    elif q_type == "who":
        #Look for person
        verb_count = 2
        noun_count = 3


    for sent in sentences:
        count = 0
        for word in sent:
            token = word[0]
            pos = word[1]
            for qword in question_words:
                if stemmer.stem(qword) == stemmer.stem(token) and token not in STOPWORDS:
                #if stemmer.stem(qword) == stemmer.stem(token):
                    if pos == "VBP" or pos == "VB":
                        count = count + verb_count
                    elif pos == "NN" or pos == "NNP" or pos == "NNS":
                        count = count + noun_count
                    else:
                        count = count + 1

        # tokenized_sentence = ' '.join([word for (word,tag) in sent])
        # print(tokenized_sentence + " " + str(count))
        best_sentences.append((count, sent))
        # best_sentences = sorted(best_sentences, key=operator.itemgetter(0), reverse=True)
        # best_sentence = (best_sentences[0])[1]    

    return best_sentences

def get_best_sentences_bow(question, story):
    qtext = get_sentences(question['text'])
    qbow = baseline.get_bow(qtext[0], STOPWORDS)
    # stext = get_sentences(story['text'])

    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])

    best_sentences = []

    # print(qbow)

    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = baseline.get_bow(sent, STOPWORDS)
        
        # print(sbow)
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)

        '''
        if overlap == 0:
            print('no overlap')
            # best_sents = get_best_sentences(question, story)
            # best_sentences.extend(best_sents)

        else:
        '''
        if overlap > 0:
            best_sentences.append((overlap, sent))

    if len(best_sentences) == 0:
        best_sentences = get_best_sentences(question, story)

    best_sentences = sorted(best_sentences, key=operator.itemgetter(0), reverse=True)
    # best_sentence = (answers[0])[1]    

    # tokenized_sent = ' '.join([word for (word, tuple) in best_sentence])
    # print((best_sentences[0])[1])
    return best_sentences

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


# def get_locations(best_sentences, chunker):

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


    """
    ###     Your Code Goes Here         ###
    '''
    chunker = nltk.RegexpParser(GRAMMAR)
    subj = get_nsubj(question)
    verb = get_verb(question)
    patterns = subj.extend(verb)
    sentences = get_sentences(story['text'])
    best_sentences = get_best_sentences([subj[0], verb], sentences)
    locations = get_locations(best_sentences, chunker)
    '''
    # print(question['text'])

    best_sentences = get_best_sentences_bow(question, story)
    best_sentence = (best_sentences[0])[1]
    tokenized_sent = ' '.join([word for (word, tuple) in best_sentence])

    if len(best_sentences) > 1:
        best_sentence2 = (best_sentences[1])[1]
        tokenized_sent2 = ' '.join([word for (word, tuple) in best_sentence2])

        tokenized_sent += tokenized_sent2

        if len(best_sentences) > 2:
            best_sentence3 = (best_sentences[2])[1]
            tokenized_sent3 = ' '.join([word for (word, tuple) in best_sentence3])

            tokenized_sent += tokenized_sent3

            if len(best_sentences) > 3:
                best_sentence4 = (best_sentences[3])[1]
                tokenized_sent4 = ' '.join([word for (word, tuple) in best_sentence4])

                tokenized_sent += tokenized_sent4

                if len(best_sentences) > 4:
                    best_sentence5 = (best_sentences[4])[1]
                    tokenized_sent5 = ' '.join([word for (word, tuple) in best_sentence5])

                    tokenized_sent += tokenized_sent5


    # print(best_sentences[0])

    answer = tokenized_sent

    ###     End of Your Code         ###
    return answer



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
