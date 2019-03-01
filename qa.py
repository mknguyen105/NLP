from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

import nltk

STOPWORDS = nltk.corpus.stopwords.words('english')

# Our simple grammar from class (and the book)
import nltk, re, operator

GRAMMAR = """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at", "behind", "below", "beside", "above", "across", "along", "below", "between", "under",
              "near", "inside"])
TIME_NN = set(
    ['today', 'yesterday', "o'clock", 'pm', 'year', 'month', 'hour', 'minute', 'second', 'week', 'after', 'before'])


# From dep demo ********************************************************************************************************
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root' or node['rel'] == 'ROOT':
            return node
    return None

def get_node_parent_siblings(node, graph):
    parent_address = node['head']
    parent = graph.nodes[parent_address]
    nodes = get_dependents(parent, graph, [])
    return nodes

def get_node_depth(node, graph):
    depth = 0
    if node['rel'] == 'root':
        return 0
    parent_address = node['head']
    parent = graph.nodes[parent_address]
    return 1 + get_node_depth(parent, graph)



# Takes a list of nodes from a dependency graph and returns the words from the tree in order
def get_subtree_phrase(node_list):
    # Sort the list by address
    sorted_nodes = sorted(node_list, key=operator.itemgetter('address'))

    # Join the words from each node
    words = []
    for node in sorted_nodes:
        words.append(node['word'])
    phrase = ' '.join(words)
    return phrase

def find_rel(graph, word, rel):
    for node in graph.nodes.values():
        if (node['word'] == word):
            # Find its rel
            if node['rel'] == rel:
                print("Rel Matched!")
                deps = get_dependents(node, graph, [])  # third parameter []
                deps = sorted(deps + [node], key=operator.itemgetter("address"))

                return " ".join(dep["word"] for dep in deps)
    return None

# Finds the nodes with the given relation in a graph
def find_node_rel(rel, graph):
    nodes = []
    for node in graph.nodes.values():
        if node['rel'] == rel:
            nodes.append(node)
    return nodes if len(nodes) > 0 else None


def find_node(word, graph):
    lmtzr = WordNetLemmatizer()
    qword = lmtzr.lemmatize(word, 'v')
    for node in graph.nodes.values():
        ntag = node["tag"]
        nword = node["word"]
        if nword is not None:
            if ntag.startswith("V"):
                nword = lmtzr.lemmatize(nword, 'v')
            else:
                nword = lmtzr.lemmatize(nword, 'n')
            if nword == qword:
                print(nword)
                return node
    return None


# Visited_nodes is a empty list
# Node in a tree
def get_dependents(node, graph, visited_nodes):  # visitednodes
    results = []

    if node in visited_nodes:
        return results
    if len(visited_nodes) == 0:
        results.append(node)
    visited_nodes.append(node)
    for item in node["deps"]:
        visited_nodes.append(item)

        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph, visited_nodes)
        # results = results + get_dependents(dep, graph)

    return results


def find_answer(qgraph, sgraph, rel):
    qmain = find_main(qgraph)
    qword = qmain["word"]

    snode = find_node(qword, sgraph)
    # If snode is none, it means it couldn't find the root word
    if snode == None:
        return None

    for node in sgraph.nodes.values():
        print("node[head]=", node["head"])
        if node["head"] == None or snode["address"] == None:
            print()
            break
        elif node.get('head', None) == snode["address"]:

            print(node["word"], node["rel"])

            if node['rel'] == rel:
                deps = get_dependents(node, sgraph, [])
                deps = sorted(deps + [node], key=operator.itemgetter("address"))

                return " ".join(dep["word"] for dep in deps)


# End of dep demo ******************************************************************************************************

# Returns the tokenized and tagged sentences from the passed text
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


# Takes a dependency graph and a given rel, like s_dep and 'nsubj', and returns the word associated with it in the graph
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


# Takes a dependency graph and a given rel, like s_dep and 'nsubj', and returns a complete phrase associated with it.
# The phrase starts at the given rel and finds all of it's dependents.
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
            deps = get_dependents(node, dep_graph, [])
            deps = sorted(deps, key=operator.itemgetter("address"))
            deps = [str(dep['word']).lower() for dep in deps]
            return " ".join(deps)


# This function returns either story_relations, or question relations
# The function iterates through all of the given deps to get from the keys of rel_score_dict and finds the word
# associated with the dependency in the graph.
def get_graph_rels(dep_graph, rel_score_dict):
    rel_dict = {}
    for relation, score in rel_score_dict.items():
        rel_dict[relation] = get_dependency_word(dep_graph, relation)
    return rel_dict


# This function takes the question dependencies and the sentence dependencies, checks for a match in q_rel, the question
# dependency, and s_rel, the story dependency and assigns a score to the match based on the value of q_rel in the
# rel_score_dict
def get_rel_score(question_relations, sentence_relations, q_rel, s_rel, rel_score_dict):
    score = 0
    if question_relations[q_rel] is not None:
        if question_relations[q_rel] == sentence_relations[s_rel]:
            score = rel_score_dict[q_rel]

    return score


# Updated this to use dep graph to find best sentences based off of relationships between the dependencies in the
# question and story sentences.
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

    # This creates tuples for each sentence in the story of the form [(sent1_dep, sent1_raw), (sent2_dep, sent2_raw)...]
    graph_sent_tuples = [(s_dep[i], sentences[i]) for i in range(len(sentences))]

    # This is the dictionary that is used to check how many points to assign for matches between the dependency in the
    # question and sentence. The keys are the dependency, and the value is the score added to the sentence when there is
    # a match of that type in the question. So if the root of the question matches the nsubj of the sentence,
    # rel_score_dict['root'] points will be added to the sentence score.
    rel_score_dict = {

    'root' :        3,
    'nmod' :        2,
    'dobj' :        2,
    'nsubj' :       2,
    'nsubjpass' :   2,
    'vmod' :        1,
    'xcomp' :       1,
    'conj' :        1,
    'advcl' :       1,
    'compound' :    1,
    'aux' :         1,
    'case' :        1,
    'cop' :         1,
    'neg' :         1,
    'cc' :          1,
    'mark' :        0

    }

    # Find all of the dependencies listed as keys in the rel score dict for the question and store them as a dictionary
    # Format is {'nsubj' : 'crow', 'root': 'sit',...}
    question_relations = get_graph_rels(q_dep, rel_score_dict)

    # Iterate through each of the sentences in the story and check for matches of various types between the sentence and
    # the story
    scored_sentences = []
    for sent_graph, sentence in graph_sent_tuples:

        score = 0

        # Find the sentence dependencies for each sentence and store in a dict just like we did for the question above
        sentence_relations = get_graph_rels(sent_graph, rel_score_dict)

        # Compare the list of all different dependency relation combinations between question and story
        for q_rel, q_relation in question_relations.items():
            for s_rel, s_relation in sentence_relations.items():
                score += get_rel_score(question_relations, sentence_relations, q_rel, s_rel, rel_score_dict)

        # Extra question specific checks

        # Create a list holding just the lowercase words in the sentence
        sentence_words = [word.lower() for (word, tag) in sentence]

        # Where
        if question_type == 'where':
            for prep in LOC_PP:
                if prep in sentence_words:
                    score += 2

        # Who


        # When
        if question_type == 'when':
            for time_word in TIME_NN:
                if time_word.lower() in sentence_words:
                    score += 4
                elif 'am' in sentence_words:
                    score += 2

        # Why
        if question_type == 'why':
            if 'because' in sentence_words:
                score += 0


        # Decision
        if question_type == 'decision':
            score += 0

        # How

        # Which


        # Add the sentence to the list of scored sentences with a final score. List contains tuples where the first val
        # is the tokenized sentence/tag tuples list, and the second val is the score of the sentence.
        # Format is a list of tuples like: [([(word1, tag1),...], 3), ([(word1, tag1),...], 2)...]
        scored_sentences.append((sentence, score, sent_graph))

    # Sort the scored sentences in desc order
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    return scored_sentences


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

    question_types = ['who', 'what', 'when', 'where', 'why', 'how', 'which']

    words = nltk.word_tokenize(question['text'])
    first_word = words[0].lower()
    if first_word in question_types:
        return first_word
    else:
        return 'decision'

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
            for pattern in ['today', 'yesterday', "o'clock", 'year', 'month', 'hour', 'minute', 'second', 'week',
                            'after', 'before']:
                candidates.extend(re.findall(pattern, sent[0]))
        answer = []
        answer = [word for word in candidates if word not in answer]

        return ' '.join(answer)


    elif question_type == 'where':
        grammar = """
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
            for word in ['because', 'so that', 'in order to', ]:
                if word in sent[0]:
                    found_words.append(word)
            for word in found_words:
                index = sent[0].index(word)
                candidates.append(sent[0][index:])
        return ' '.join(candidates)

    return ''


def narrow_answer(q_type, q_dep, sent_dep, answer):
    # Find root
    q_root = find_main(q_dep)
    sent_root = find_main(sent_dep)

    # Varibles
    q_root_word = q_root['word']
    sent_root_word = q_root['word']

    # Nsubj of Sentence dependency
    q_nsubj_root = get_dependency_word(q_dep, 'nsubj')
    # Dobj of Sentence Dependency
    s_dobj_root = get_dependency_word(sent_dep, 'dobj')
    # Subject of Sentence Dependency
    subj_word_sent = get_dependency_word(sent_dep, 'nsubj')
    # nmod of sentence dependency
    sent_nmod = get_dependency_word(sent_dep, 'nmod')
    # det of sentence dependency (ex. The )
    sent_det = get_dependency_word(sent_dep, 'det')
    # nmod phrase in sentence dependency
    sent_nmod_phrase = get_dependency_phrase(sent_dep, 'nmod')

    print("Sentence Root Word Is: " + sent_root_word)
    print("Question Root Word Is: " + q_root_word)

    if q_type == "who":

        # Check if subj has a conjunction
        extension = find_rel(sent_dep, subj_word_sent, 'nmod')
        # Find nsubj
        extension2 = get_dependency_phrase(sent_dep, 'nsubj')

        # Special Case: If the Root word contains an nsubj thats not a question type. Ex: Who did the fox invite?

        if q_nsubj_root and s_dobj_root and q_nsubj_root != q_type:
            print("S_Dobj_Root: " + str(s_dobj_root))  # stork. Make it #Stork
            # extension3 = find_rel(sent_dep, s_dobj_root, 'det')
            extension3 = get_dependency_word(sent_dep, 'det')

            if extension3 is not None:
                answer = extension3 + " " + s_dobj_root
            else:
                answer = s_dobj_root

            return answer

        # Adds it to the answer in order

        if extension2:
            answer = extension2 + " " + subj_word_sent
        else:
            answer = subj_word_sent

        if extension:
            answer = answer + " " + extension + " " + subj_word_sent

        return answer

    elif q_type == "what":

        """
        if q_nsubj_root != q_type and q_nsubj_root:
            #Find the last nmod to improve precision?
            before_keyword, keyword, after_keyword = answer.partition(sent_root_word)
            if after_keyword is not None:
                print("After KeyWord is " + str(after_keyword))
                answer = after_keyword
                return answer
            else:
                return answer
        """

        if sent_det is not None and sent_nmod is not None:
            answer = str(sent_det) + " " + str(sent_nmod)
        elif sent_nmod is not None:
            answer = sent_nmod

        return answer

    elif q_type == "when":
        # Do something

        return answer

    elif q_type == "where":
        # Do something

        return answer

    elif q_type == "why":
        # Do something
        node = find_node('because', sent_dep)
        if node is not None:
            node_fam = get_node_parent_siblings(node, sent_dep)
            answer = get_subtree_phrase(node_fam)
            conj = find_node_rel('conj', sent_dep)
            cc = find_node_rel('cc', sent_dep)
            #if cc is not None:
            #    phrase = get_dependency_phrase(sent_dep, 'cc')
            #    answer += ' ' + phrase
            #if conj is not None:
            #    phrase = get_dependency_phrase(sent_dep, 'conj')
            #    answer += ' ' + phrase

        else:
            marks = find_node_rel('mark', sent_dep)
            answer = ''
            if len(marks) > 1:
                #min_depth = 1000000
                #shallow_node = None
                nodes = []
                for mark in marks:
                # Depth solution doesn't work due to incorrect parse tree
                    #depth = get_node_depth(mark, sent_dep)
                    #if min_depth < depth:
                    #   min_depth = depth
                    #   shallow_node = mark

                    # Hard coding best solution with current code setup and bad parse
                    if mark['word'] != 'while':
                        nodes.extend(get_node_parent_siblings(mark, sent_dep))
                        answer = get_subtree_phrase(nodes)


            elif len(marks) == 1:
                node_fam = get_node_parent_siblings(marks[0], sent_dep)
                answer = get_subtree_phrase(node_fam)

            return answer

        return answer
    elif q_type == "decision":
        answer = "yes no"
        # Either yes or no
        return answer

    return answer


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
    print(answer + '\t' + str(best_sentence_score) + " " + question['qid'] + '\n')
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()
    # candidates = get_candidates(question, story, best_sentences)
    # answer = candidates

    # Take the top three sentences and join them together to increase recall before searching for an answer
    # answer = [raw_sent for (raw_sent, sent, count) in best_sentences[0:1]]
    # answer = ' '.join(answer)
    # answer = get_sentence(best_sentences)
    sent_dep = best_sentences[0][2]

    narrowed_answer = narrow_answer(question_type, q_dep, sent_dep, answer)
    return narrowed_answer

    # return answer

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
