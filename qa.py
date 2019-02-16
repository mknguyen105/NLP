
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk, re

def get_tree_words(root):
    sent = []
    for node in root:
        if type(node) == nltk.Tree:
            sent += get_tree_words(node)
        elif type(node) == str:
            sent.append(node)
    return sent


# Lemmatizes word and lowercases. Then generates a regex pattern that matches the lemmatized word to other forms of the
# word. So Shipping generates a pattern that will match ship, shipping, Shipping, ships, Ships, shiply and Shiply
def get_word_pattern(word):
    lmtzr = nltk.stem.WordNetLemmatizer()
    word = lmtzr.lemmatize(word.lower())
    pattern = '\b([' + word[0].upper() + word[0].lower() + ']' + word[1:] +'(ly|(' + word[-1:] + ')ing|s)?)\b'
    return pattern

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

# Add other methods of detecting question type
    # Who

    # What

    # When

    # Where

    # Why

    # How


# A better solution may be to use the dependency parse of the question deconstruct the subject direct obj etc..., and
# then use a regex to find them in the text. So if subj is Crow and direct obj is branch, search for ^\..*(Crow.).*(branch)$

# Naive solution for best sentences
# Add lemmatizer
# This function will return a list of tuples of sentences in the story that contain non-stopwords from the question
# first elm is the sentence, second elm is number of overlapping words. Returns in desc sorted order of overlaps.
def get_best_sentences(question, story):
    story_text = story['text']
    question_text = question['text']
    lmtzr = nltk.stem.WordNetLemmatizer()

    question_words = nltk.word_tokenize(question_text)
    sentences = nltk.sent_tokenize(story_text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    question_words = [get_word_pattern(word) for word in question_words if word.lower() not in nltk.corpus.stopwords.words('english') and word.isalpha()]


    # Iterate through each word of each sentence in the story, checking if words from the question are in the sentence
    # If there is a match, add the sentence to best sentences, and join the words of the sentences before returning them
    best_sentences = {}
    ranked_sentences = []

    for pattern in question_words:
        for sent in sentences:
            overlaps = re.findall(pattern, sent)
            if len(overlaps) > 0:
                if sent not in best_sentences:
                    best_sentences[sent] = len(overlaps)
                else:
                    best_sentences[sent] += len(overlaps)

    for sent in sentences:
        for word in sent:
            if word.lower() in question_words:
                if ' '.join(sent) not in best_sentences:
                    best_sentences[' '.join(sent)] = 1
                else:
                    best_sentences[' '.join(sent)] += 1
    for (key, value) in best_sentences.items():
        ranked_sentences.append((key, value))
    ranked_sentences.sort(key=lambda x: x[1], reverse=True)
    return best_sentences


# Iterates through the constituency tree, checking for a given pattern of parts of speech, returning a subtree
# containing it if found, else, returns none
def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None


def get_likely_answers(question, story):
    tree = story["sch_par"][1]
    question_type = get_question_type(question)

    if question_type == 'where':
        pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")

    subtree = pattern_matcher(pattern, tree)
    sub_sentence = get_tree_words(subtree)[1:]
    answer = ' '.join(sub_sentence)
    print(answer)


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
    sentences = get_best_sentences(question, story)
    print(sentences)
    answer = "whatever you think the answer is"

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
