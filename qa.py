
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from textblob import TextBlob, Word

import nltk
STOPWORDS = nltk.corpus.stopwords.words('english')

# Our simple grammar from class (and the book)
import nltk, re, operator

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
TIME_NN = set(['today', 'yesterday', "o'clock", 'year', 'month', 'hour', 'minute', 'second', 'week', 'after', 'before'])

# From dep demo ********************************************************************************************************
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root' or node['rel'] == 'ROOT':
            return node
    return None


def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None


def get_dependents(node, graph, visited_nodes):
    results = []
    if node in visited_nodes:
        return results
    visited_nodes.append(node)
    for item in node["deps"]:
        visited_nodes.append(item)
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph, visited_nodes)

    return results

def find_answer(qgraph, sgraph, rel):
    qmain = find_main(qgraph)
    qword = qmain["word"]

    snode = find_node(qword, sgraph)

    for node in sgraph.nodes.values():
        # print("node[head]=", node["head"])
        if node.get('head', None) == snode["address"]:
            # print(node["word"], node["rel"])

            if node['rel'] == rel:
                deps = get_dependents(node, sgraph)
                deps = sorted(deps + [node], key=operator.itemgetter("address"))

                return " ".join(dep["word"] for dep in deps)
# End of dep demo ******************************************************************************************************

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def get_dependency_word(dep_graph, rel):
    lmtzr = WordNetLemmatizer()
    for node in dep_graph.nodes.values():
        if node['rel'] == rel:
            word = str(node['word']).lower()
            if node['tag'][0] == 'V':
                word = lmtzr.lemmatize(word, 'v')
            elif node['tag'][0] == 'N':
                word = lmtzr.lemmatize(word, 'n')
            return word.lower()
    return None

def get_dependency_phrase(dep_graph, rel):
    lmtzr = WordNetLemmatizer()
    if rel == 'root':
        root = find_main(dep_graph)
        if root['tag'][0] == 'V':
            word = lmtzr.lemmatize(root['word'], 'v')
        elif root['tag'][0] == 'N':
            word = lmtzr.lemmatize(root['word'], 'n')
        return word
    for node in dep_graph.nodes.values():
        if node['rel'] == rel:
            deps = get_dependents(node, dep_graph, visited_nodes=[])
            deps = sorted(deps, key=operator.itemgetter("address"))
            deps = [str(dep['word']).lower() for dep in deps]
            return " ".join(deps)

def get_graph_rels(dep_graph, rel_score_dict):
    rel_dict = {}
    for relation, score in rel_score_dict.items():
        rel_dict[relation] = get_dependency_word(dep_graph, relation)
    return rel_dict

def get_rel_score(question_relations, sentence_relations, q_rel, s_rel, rel_score_dict):
    score = 0
    if question_relations[q_rel] is not None:
        if question_relations[q_rel] == sentence_relations[s_rel]:
            score = rel_score_dict[q_rel]

    return score



# This should ultimately be redone using regex so that we can match longer chains of words like bigrams, trigrams etc...
def get_best_sentences(q_dep, s_dep, sentences, question_type):

    """
    Split question type, and find it through the dependency graph

    What
        If question is asking for a noun:
            Find the root word of question and sentence, then find its nmod (root's dependency') and nsubj
        If question is asking for a verb:
            Find the nsubj & nmod (overlapping ex. cheese). Follow up the leaves, and theyre dependent on the question verb
        
    Who
         Find root
         Find subj dependent of that root
    Where
        

    Why
    
    """""

    graph_sent_tuples = [(s_dep[i], sentences[i]) for i in range(len(sentences))]

    rel_score_dict = {
    'root' : 3,
    'nmod' : 2,
    'dobj' : 2,
    'nsubj' : 1,
    'nsubjpass' : 1,
    'nsubj' : 1,
    'xcomp' : 1,
    'conj' : 1,
    'advcl' : 1,
    'compound' : 1
    }

    question_relations = get_graph_rels(q_dep, rel_score_dict)

    scored_sentences = []
    for sent_graph, sentence in graph_sent_tuples:
        score = 0
        sentence_relations = get_graph_rels(sent_graph, rel_score_dict)
        sentence_words = [word.lower() for (word, tag) in sentence]

        # Check one to one comparisons like question nsubj vs sentence nsubj
        for rel, relation in question_relations.items():
            score += get_rel_score(question_relations, sentence_relations, rel, rel, rel_score_dict)

        # Check other comparisons that might be relevant like question nsubj to sentence nsubjpass
        score += get_rel_score(question_relations, sentence_relations, 'nsubj', 'nsubjpass', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'dobj', 'nmod', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'nmod', 'dobj', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'root', 'xcomp', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'root', 'conj', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'conj', 'advcl', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'advcl', 'conj', rel_score_dict)
        score += get_rel_score(question_relations, sentence_relations, 'compound', 'nsubj', rel_score_dict)



        if question_type == 'where':
            for prep in LOC_PP:
                if prep in sentence_words:
                    score += 1


        if question_type == 'why':
            if 'because' in sentence_words:
                score += 2


        scored_sentences.append((sentence, score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return scored_sentences




# LIKELY DELETE***************
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

# This will find all of the subjects of a story or question
def find_subjects(dep):
    subjects = []
    i = 0
    if type(dep) == list:
        for sentence in dep:
            dep_graph = sentence
            while dep_graph.contains_address(i):
                if dep_graph._rel(i) == 'nsubj':
                    word = dep_graph.get_by_address(i)['word']
                    if word not in subjects:
                        subjects.append(word.lower())
                i += 1
    else:
        dep_graph = dep
        while dep_graph.contains_address(i):
            if dep_graph._rel(i) == 'nsubj':
                word = dep_graph.get_by_address(i)['word']
                if word not in subjects:
                    subjects.append(word.lower())
            i += 1

    return subjects

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
# ****************************


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

def get_tree_words(root):
    sent = []
    for node in root:
        if type(node) == nltk.Tree:
            sent += get_tree_words(node)
        elif type(node) == tuple:
            sent.append(node[0])
    return sent


# LIKELY DELETE***********************************************
# If two verbs are similar, this will return an integer that adds to the overlap score. So feel and felt will have
# the same meaning in some contexts, so it will return whatever value it's set to return on match
def get_verb_similarity(verb1, verb2):
    score = 0
    try:
        word1 = wn.synsets(verb1, pos='v')
        word2 = wn.synsets(verb2, pos='v')
        for syn1 in word1:
            for syn2 in word2:
                if syn1.path_similarity(syn2) == 1:
                    return 2
    except:
        return 0

    return score
#*************************************************************


# Increase precision by locating where in the best sentences the answer might be
def get_candidates(question, story, best_sentences):
    candidates = []
    question_type = get_question_type(question)
    qverb = get_verb(question)
    qsub = find_subjects(question['dep'])
    qwords = nltk.word_tokenize(question['text'])
    qtags = nltk.pos_tag(qwords)
    story_subjects = find_subjects(story['story_dep'])
    lmtzr = WordNetLemmatizer()

    if question_type == 'who':
        possible_answers = story_subjects
        answer = ''
        if type(qsub) == list and len(qsub) > 0:
            if qsub[0] == 'story':
                answer = 'A ' + story_subjects[0]
                for subj in story_subjects[1:]:
                    answer += ' and a ' + subj
        else:
            return ' '.join(story_subjects)
        return answer


    elif question_type == 'what':

        answer = [raw_sent for (raw_sent, sent, count) in best_sentences[0:2]]
        answer = ' '.join(answer)
        return answer

    elif question_type == 'when':
        for sent in best_sentences:
            for pattern in ['today', 'yesterday', "o'clock", 'year', 'month', 'hour', 'minute', 'second', 'week', 'after', 'before']:
                candidates.extend(re.findall(pattern, sent[0]))
        answer = []
        answer = [word for word in candidates if word not in answer]

        return ' '.join(answer)


    elif question_type == 'where':
        grammar =   """
                    N: {<PRP>|<NN.*>}
                    ADJ: {<JJ.*>}
                    NP: {<DT>? <ADJ >* <N>+}
                    PP: {<IN> <NP> <IN>? <NP>?}
                    """
        chunker = nltk.RegexpParser(grammar)
        if len(qsub) > 0:
            subj = lmtzr.lemmatize(qsub[0], 'n')
        else:
            subj = story_subjects[0]
        verb = lmtzr.lemmatize(qverb, 'v')

        for sent in best_sentences:

            # If the verb and subject are in the sentence, use this solution only
            if subj in sent[0] or verb in sent[0]:
                tree = chunker.parse(sent[1])
                locations = find_locations(tree)
                if len(locations) > 0:
                    locations = get_tree_words(locations)
                    candidates = locations
                    break

            # If a sent isn't found where subj and verb are in the solution, use all sentences locations
            else:
                tree = chunker.parse(sent[1])
                locations = find_locations(tree)
                if len(locations) > 0:
                    locations = get_tree_words(locations)
                candidates.extend(locations)


        answer = ' '.join(candidates)
        return answer

    elif question_type == 'why':
        for sent in best_sentences:
            found_words = []
            for word in ['because', 'so that', 'in order to',]:
                if word in sent[0]:
                    found_words.append(word)
            for word in found_words:
                index = sent[0].index(word)
                candidates.append(sent[0][index:])
        return ' '.join(candidates)

    return ''

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

    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
        s_dep = story['story_dep']
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])
        s_dep = story['sch_dep']
    q_dep = question['dep']
    question_type = get_question_type(question)

    print(question['text'])
    best_sentences = get_best_sentences(q_dep, s_dep, sentences, question_type)
    best_sentence_text = [word for (word, tag) in best_sentences[0][0]]
    best_sentence_score = best_sentences[0][1]
    answer = ' '.join(best_sentence_text)
    print(answer + '\t'  + str(best_sentence_score) + '\n')
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
    #candidates = get_candidates(question, story, best_sentences)
    #answer = candidates


    # Take the top three sentences and join them together to increase recall before searching for an answer
    #answer = [raw_sent for (raw_sent, sent, count) in best_sentences[0:1]]
    #answer = ' '.join(answer)
    #answer = get_sentence(best_sentences)

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


def run_qa(evaluate=False):
    QA = QAEngine()
    QA.run()
    QA.save_answers()

#############################################################

def main():
    run_qa(evaluate=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
