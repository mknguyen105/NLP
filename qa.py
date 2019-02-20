
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
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
    lower_sentences = []
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for sent in sentences:
        lower_sentence = []
        for word in sent:
            lower_sentence.append(word.lower())
        lower_sentences.append(lower_sentence)
    lower_sentences = [nltk.pos_tag(sent) for sent in lower_sentences]
    return lower_sentences

def get_best_sentences(patterns, sentences):
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        matches = 0
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                continue
            else:
                matches += 1
        if matches > 0:
            result.append((sent, matches))

    # Sort list of best sentences in descending order from most overlaps to least
    result.sort(key=lambda x: x[1], reverse=True)

    # Create a list of results without the counts and return the list
    results = [res for (res, count) in result]

    return result


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


def get_nouns(dep):
    nouns = []
    i = 0
    dep_graph = dep['dep']
    while dep_graph.contains_address(i):
        if dep_graph._rel(i) == 'nsubj' or dep_graph._rel(i) == 'nmod':
            word = dep_graph.get_by_address(i)['word']
            nouns.append(word)
        i += 1

    return nouns



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


# This function goes through all sentences given to it, and finds chunks that correspond to the question type and
# returns them as a list. The function iterates through each list, creates the chunks, and then finds all patterns
# that correspond to that question. For example, if the question is a where question, this will search through the
# chunks and add the ones that start with a PP
def find_candidates(sentences, chunker, question_type):
    candidates = []

    for sent in sentences:
        tree = chunker.parse(sent)
        # print(tree)
        if question_type == 'where':
            locations = find_locations(tree)
            candidates.extend(locations)
        elif question_type == 'what':
            subjects = find_subjects(tree)
            candidates.extend(subjects)

    return candidates

def np_filter(subtree):
    return subtree.label() == "NP"

def pp_filter(subtree):
    return subtree.label() == "PP"


def is_location(prep):
    return prep[0] in LOC_PP


def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations

    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    results = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)

    if len(locations) > 0:
        results.append(' '.join(get_tree_words(locations[0])))
    return results

def find_subjects(tree):
    subjects = []
    results = []
    for subtree in tree.subtrees(filter=np_filter):
        subjects.append(subtree)

    results.append(get_tree_words(subjects[0]))
    return results

def get_tree_words(root):
    sent = []
    for node in root:
        if type(node) == nltk.Tree:
            sent += get_tree_words(node)
        elif type(node) == tuple:
            sent.append(node[0])
    return sent

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

    # Determine where we will find the answer
    if question['type'] == 'Sch':
        text = story['sch']
        dep = story['sch_dep']
        par = story['sch_par']
    else:
        text = story['text']
        dep = story['story_dep']
        par = story['story_par']

    # Setup a chunker to be used later to find parts of the sentence that may contain the answer
    chunker = nltk.RegexpParser(GRAMMAR)

    # Get the question type. Returns a string of 'who', 'what', 'where', etc...
    question_type = get_question_type(question)

    # Get the verb and subject of the question and put them into the list 'patterns'. Ex:['crow', fox, 'sit']
    nouns = get_nouns(question)
    verb = get_verb(question)
    patterns = [noun for noun in nouns]
    patterns.append(verb)

    # A list of tokenized and tagged sentences from the story. Format is [[(word1, tag1), (word2, tag2),...],[word1,..]]
    sentences = get_sentences(text)

    # Get the best sentences, found by scoring words in sentences that overlap with the subj and verb in the question
    best_sentences = get_best_sentences(patterns, sentences)

    # Go through the best sentences and find parts of the sentences that relate to the question type
    if len(best_sentences) > 0:
        candidates = find_candidates(best_sentences, chunker, question_type)

    # Just return the first result for now
        answer = candidates
    else:
        answer = ''


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
