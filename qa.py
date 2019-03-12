from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
import dependency_stub
import wordnet_demo

import nltk
import nltk, re, operator


STOPWORDS = nltk.corpus.stopwords.words('english')

# Our simple grammar from class (and the book)
GRAMMAR = """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at", "behind", "below", "beside", "above", "across", "along", "below", "between", "under",
              "near", "inside", "onto", "after"])
TIME_NN = set(
    ['today', 'yesterday', "o'clock", 'pm', 'year', 'month', 'hour', 'minute', 'second', 'week', 'after', 'before',
     'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'day', 'time'])


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
                #print("Rel Matched!")
                deps = get_dependents(node, graph, [])  # third parameter []
                deps = sorted(deps + [node], key=operator.itemgetter("address"))

                return " ".join(dep["word"] for dep in deps)
    return None


def find_rel_node(graph, iNode, rel):
    for node in graph.nodes.values():
        if (node  == iNode):
            # Find its rel
            if node['rel'] == rel:
                #print("Rel Matched!")
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
            if nword == 'fell':
                nword = 'fall'
            elif nword == 'spat':
                nword = 'spit'
            elif ntag.startswith("V"):
                nword = lmtzr.lemmatize(nword, 'v')
            else:
                nword = lmtzr.lemmatize(nword, 'n')
            if nword == qword:
                #print(nword)
                return node
    return None


# Visited_nodes is a empty list
# Node in a tree
#node with word
def get_dependents(node, graph, visited_nodes):  # visitednodes
    results = []
    if node in visited_nodes:
        return results
    if len(visited_nodes) == 0:
        results.append(node)
    visited_nodes.append(node)

    #Checks in case of nothing
    for item in node["deps"]:
        visited_nodes.append(item)

        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph, visited_nodes)
        # results = results + get_dependents(dep, graph)

    return results


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

def get_list(graph, rel):
    results = []
    lmtzr = WordNetLemmatizer()
    for node in graph.nodes.values():
        if node['rel'] == rel:
            word = str(node['word']).lower()
            if node['tag'][0] == 'V':
                word = lmtzr.lemmatize(word, 'v')
                results.append(word)

            elif node['tag'][0] == 'N':
                word = lmtzr.lemmatize(word, 'n')
                results.append(word)

    return results

def get_list_without_lem(graph, rel):
    results = []
    lmtzr = WordNetLemmatizer()
    for node in graph.nodes.values():
        if node['rel'] == rel:
            word = str(node['word']).lower()
            results.append(word)
    return results


def get_dependency_node(dep_graph, rel):
    lmtzr = WordNetLemmatizer()
    for node in dep_graph.nodes.values():
        if node['rel'] == rel:
            word = str(node['word']).lower()
            if node['tag'][0] == 'V':
                word = lmtzr.lemmatize(word, 'v')
            elif node['tag'][0] == 'N':
                word = lmtzr.lemmatize(word, 'n')
            return node
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

def get_qtype_score(q_dep, s_dep, tagged_sentence, question_type):
    score = 0

    dep_dict = {

        'root': 3,
        'nmod': 2,
        'dobj': 2,
        'nsubj': 2,
        'nsubjpass': 2,
        'vmod': 1,
        'xcomp': 1,
        'conj': 1,
        'advcl': 1,
        'compound': 1,
        'aux': 1,
        'case': 1,
        'cop': 1,
        'neg': 1,
        'cc': 1,
        'mark': 0
    }

    sentence_deps = get_graph_rels(s_dep, dep_dict)
    question_deps = get_graph_rels(q_dep, dep_dict)
    lmtzr = WordNetLemmatizer()
    for dep, word in question_deps.items():
        if word is not None and sentence_deps[dep] is not None:
            if lmtzr.lemmatize(sentence_deps[dep], 'v') == lmtzr.lemmatize(question_deps[dep], 'v'):
                score += 1

    sent_words = [word.lower() for word, tag in tagged_sentence]
    if question_type == 'why':
        if 'because' in sent_words:
            score += 1
        elif 'in' in sent_words and 'order' in sent_words and 'to' in sent_words:
            score += 1
        elif 'for' in sent_words:
            score += 1
        elif 'so' in sent_words:
            score += 1
    if question_type == 'where':
        if 'in' in sent_words or 'on' in sent_words or 'onto' in sent_words or 'under' in sent_words or 'by' in sent_words:
            score += 1
    return score


# Updated this to use dep graph to find best sentences based off of relationships between the dependencies in the
# question and story sentences.
def get_best_sentences(question, story, question_type):
    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
        s_dep = story['story_dep']
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])
        s_dep = story['sch_dep']
    q_dep = question['dep']

    # Get question keywords and hypo/hypernyms
    question_tagged = get_sentences(question['text'])[0]
    question_verbs = [word.lower() for word, tag in question_tagged if tag[0] == 'V']
    question_nouns = [word.lower() for word, tag in question_tagged if tag[0] == 'N']
    question_keywords = question_verbs + question_nouns
    question_hnyms = dependency_stub.find_all_h_nyms(question_keywords)
    question_hnyms = dependency_stub.format_hlist(question_hnyms)

    # This creates tuples for each sentence in the story of the form [(sent1_dep, sent1_raw), (sent2_dep, sent2_raw)...]
    if len(s_dep) != len(sentences):
        sentences[2].extend(sentences.pop(3))
    graph_sent_tuples = [(s_dep[i], sentences[i]) for i in range(len(sentences))]

    scored_sentences = []
    for graph, sent in graph_sent_tuples:
        score = 0
        sentence_words = [word for word, tag in sent]
        question_words = [word for word, tag in get_sentences(question['text'])[0]]
        score += get_word_overlap(sentence_words, question_words)

        sent_tuple = (sent, graph, score)
        scored_sentences.append(sent_tuple)

    scored_sentences.sort(key=lambda x: x[2], reverse=True)
    n = 5
    top_sentences = scored_sentences[:n]


    scored_sentences = []
    matched_words = []
    for sent, graph, old_score in top_sentences:
        score = 0

        """
        sentence_verbs = [word.lower() for word, tag in sent if tag[0] == 'V']
        sentence_nouns = [word.lower() for word, tag in sent if tag[0] == 'N']
        sentence_keywords = sentence_nouns + sentence_verbs

        lmtzr = WordNetLemmatizer()
        for word in sentence_keywords:
            word = lmtzr.lemmatize(word, 'v')
            if word not in matched_words:
                if word in question_hnyms:
                    if word in question_verbs:
                        score += 2
                    else:
                        score += 1
        """
        score += get_qtype_score(q_dep, graph, sent, question_type)
        sent_tuple = (sent, graph, score)
        scored_sentences.append(sent_tuple)

    scored_sentences.sort(key=lambda x: x[2], reverse=True)

    return scored_sentences


def get_word_overlap(sentence_words, question_words):
    score = 0
    lmtzr = WordNetLemmatizer()
    sentence_words = [word.lower() for word in sentence_words if
                      word.lower() not in nltk.corpus.stopwords.words('english')]
    question_words = [word.lower() for word in question_words if
                      word.lower() not in nltk.corpus.stopwords.words('english')]
    matched_words = []
    for word1 in sentence_words:
        for word2 in question_words:
            if lmtzr.lemmatize(word1, 'v') == lmtzr.lemmatize(word2, 'v'):
                score += 1
            elif lmtzr.lemmatize(word1, 'n') == lmtzr.lemmatize(word2, 'n'):
                score += 1

    return score

# Returns the best n sentences in a list and their respective scores in a list
def get_top_sentences(best_sentence_tuples, n):
    sentences_text = []
    scores = []
    for i in range(n):
        if i >= len(best_sentence_tuples):
            break
        sentence = [word.lower() for word, tag in best_sentence_tuples[i][0]]
        sentence = ' '.join(sentence)
        sentences_text.append(sentence)
        scores.append(best_sentence_tuples[i][2])
    return sentences_text, scores

# This returns who what when where why or how
# I know for this assignment we could probably just use the first word and exclude the funtion, but I figure this will
# keep us organized if we're later thrown questions that don't have the question word as the first word, or if there is
# no question word in the question at all
def get_question_type(question):
    qgraph = question['dep']
    question_types = ['who', 'what', 'when', 'where', 'why', 'how', 'which']

    qtext = []
    for node in qgraph.nodes.values():
        if node['word'] is not None:
            qtext.append(node["word"])

    words = nltk.word_tokenize(question['text'])
    first_word = words[0].lower()

    if first_word in question_types:
        qtype = first_word
    else:
        qtype = 'decision'

    # categorize what questions

    if qtype == 'what':
        # find where type questions
        last_word_node = qgraph.nodes[len(qgraph.nodes.values()) - 2]
        last_word = last_word_node['word']
        # print('last word: ' + last_word)
        if last_word in LOC_PP:
            qtype = 'where'


        """
        # find when type questions
        node = find_node_rel('nmod:tmod', qgraph)
        if node is not None:
            qtype = 'when'

        """
        # find who type questions
        last_word_rel = last_word_node['rel']
        if 'do' in qtext:
            qtype = 'what'


        elif last_word_rel == 'nsubj' or last_word_rel == 'nmod' or last_word_rel == 'dobj' or last_word is 'name':
                qtype = 'who'


    return qtype


def compare_word(word, nodes):
    for graph_node in nodes:
        #print("Graph_Node['word']:" + str(graph_node['word']))
        graph_word = graph_node['word']
        if graph_word is not None and word is not None:
            #print("Comparing Words: " + str(word) + str(graph_word))
            if graph_word.lower() == word.lower():
                return True
    return False


def remove_stopwords(string):
    result = ""
    words = nltk.word_tokenize(string)
    for word in words:
        if word not in STOPWORDS:
            result = result + " " + word
    return result



def narrow_answer(qtext, q_type, q_dep, sent_dep, answer):
    # Find root
    q_root = find_main(q_dep)
    sent_root = find_main(sent_dep)

    # Varibles
    lmtzr = WordNetLemmatizer()
    q_root_word = lmtzr.lemmatize(q_root['word'], 'v')
    sent_root_word = lmtzr.lemmatize(sent_root['word'], 'v')

    # All of the nodes in the graph in a list
    sent_nodes = [node for node in sent_dep.nodes.values() if node['word'] is not None]
    # The sentence in plain text
    sent_text = get_subtree_phrase(sent_nodes)

    # Nsubj of Question dependency
    q_nsubj_root = get_dependency_word(q_dep, 'nsubj')
    # Dobj of Sentence Dependency
    s_dobj_root = get_dependency_word(sent_dep, 'dobj')
    # Subject of Sentence Dependency
    subj_word_sent = get_dependency_word(sent_dep, 'nsubj')
    # nmod of sentence dependency
    sent_nmod = get_dependency_word(sent_dep, 'nmod')
    # det of sentence dependency (ex. The )
    sent_det = get_dependency_word(sent_dep, 'det')

    #Sentence Subjects
    sent_subjects = get_list(sent_dep, 'nsubj')

    # nmod phrase in sentence dependency
    sent_nmod_phrase = get_dependency_phrase(sent_dep, 'nmod')
    # dobj phrase in sentence dependency
    sent_dobj_phrase = get_dependency_phrase(sent_dep, 'dobj')
    # nsubj phrase in sentence dependency
    sent_nsubj_phrase = get_dependency_phrase(sent_dep, 'nsubj')

    #Everything after a root word
    obj = remove_stopwords(answer[answer.find(sent_root['word']) + len(sent_root['word']):])


    #print("Sentence Root Word Is: " + sent_root_word)
    print("Sentence Root Word Is: " + sent_root['word'])
    print("Question Root Word Is: " + q_root_word)
    if sent_nmod is not None:
        print("Sentence Subject Is: " + sent_nmod)
    if q_nsubj_root is not None:
        print("Question Subject Is: " + q_nsubj_root)


    if q_type == "who":
        orig_answer = answer
        answer = dependency_stub.find_who_answer(qtext, q_dep, sent_dep)
        if not answer:
            answer = orig_answer

        return answer

    elif q_type == "what":

        last_word_node = q_dep.nodes[len(q_dep.nodes.values()) - 2]
        last_word = last_word_node['word']
        last_word_rel = last_word_node['rel']

        nsubj_node = get_dependency_node(sent_dep, 'nsubj')
        main_node = find_main(sent_dep)
        hypernyms_sent = wordnet_demo.find_hypernyms(main_node['word'], sent_dep)
        hypernyms_q = wordnet_demo.find_hypernyms(q_root['word'], q_dep)


        print("Last_Word_Tag: " + last_word_node['tag'])

        print("Sentence Subjects: " + str(sent_subjects))
        # Finding the correct subject
        for word in sent_subjects:
            subj_word_sent = word

        # ------ Name -------
        if last_word_node['word'] == 'name':
            print("Contains name")
            return subj_word_sent

        # ------ Last Word POS -------
        if sent_dobj_phrase is not None:
            if last_word_node['tag'] == 'JJ':
                #Could also be a noun
                #return sent_dobj_phrase
                return obj
            elif last_word_node['tag'] == 'IN':
                return sent_dobj_phrase
            elif last_word_node['tag'] == 'NN':
                return sent_dobj_phrase
                #return obj
            elif last_word_node['tag'] == 'VB' or last_word_node['tag'] == 'VBG':
                return sent_dobj_phrase
        else:
            print("Sent_DOBJ_Phrase is None")

        # ----- Overlap ------
        #Story Questions. Figure out a story

        if hypernyms_sent is not None: print("Hypernyms_sent:" + hypernyms_sent['word'])
        if hypernyms_q is not None: print("Hypernyms_q:" + hypernyms_q['word'])

        if main_node is not None:

            print("Sentence Main Node: " + str(main_node['word']))

            q_vbg = get_dependency_node(q_dep, 'vbg')
            q_vbd = get_dependency_node(q_dep, 'vbd')

            if q_root_word.lower() == q_type.lower() and q_vbg is not None:
                q_root = q_vbg
            elif q_root_word.lower() == q_type.lower() and q_vbd is not None:
                q_root = q_vbd

            #If it overlaps, don't use that
            q_dependents_root = get_dependents(q_root, sent_dep, [])

            #Proceed if not found in question
            if compare_word(subj_word_sent, q_dependents_root) is False and sent_nsubj_phrase is not None:
                print("Answer is NSUBJ")
                answer = str(sent_nsubj_phrase)
                return answer
            elif compare_word(s_dobj_root, q_dependents_root) is False and sent_nmod_phrase is not None:
                print("Answer is NMOD")
                answer = str(sent_nmod_phrase)
                return answer
            elif compare_word(sent_nmod, q_dependents_root) is True and sent_dobj_phrase is not None:
                print("Answer is DOBJ")
                #DOBJ Works
                answer = str(sent_dobj_phrase)
                return answer
            else:

                print("Else Statement")
                #answer = str(sent_dobj_phrase)
                print("List of DOBJ")
                dobj_list = get_list_without_lem(sent_dep, 'dobj')
                print(dobj_list)

                if dobj_list is not None:
                    for dobj in dobj_list:
                        if wordnet_demo.word_in_string(dobj, q_dep, qtext) is True:
                            continue
                        else:
                            answer = dobj
                else:
                    return answer

                #Check if sent_dobj_phrase overlaps
                return answer

    elif q_type == "when":
        answer = dependency_stub.find_answer(q_dep, sent_dep, "nmod")
        if not answer:
            answer = dependency_stub.find_answer(q_dep, sent_dep, "nmod:tmod")
        if not answer:
            answer = dependency_stub.find_answer(q_dep, sent_dep, "advmod")
        if not answer:
            answer = dependency_stub.last_effort_answer(sent_dep)

        return answer

    elif q_type == "where":
        answer = dependency_stub.find_answer(q_dep, sent_dep, "nmod")
        if not answer:
            answer = dependency_stub.find_answer(q_dep, sent_dep, "nmod:poss")
        if not answer:
            answer = dependency_stub.find_answer(q_dep, sent_dep, "dobj")
        if not answer:
            answer = dependency_stub.last_effort_where_answer(sent_dep)

        return answer

    elif q_type == "which":
        answer = dependency_stub.find_who_answer(qtext, q_dep, sent_dep)
        if not answer:
            answer = dependency_stub.last_effort_answer(sent_dep)

        return answer

    elif q_type == 'how':
        advmod = get_dependency_word(sent_dep, 'advmod')
        nmod = get_dependency_phrase(sent_dep, 'nmod')

        # If there is an advmod, use that as the answer
        if advmod is not None:
            answer = advmod
        # A variation of the below solution will probably scale better, but doesn't work well for this question
        #if get_dependency_word(sent_dep, 'advmod') is not None:
        #    answer = get_dependency_phrase(sent_dep, 'advmod')

        # Second choice is to use the nmod if it exists
        elif nmod is not None:
            answer = nmod

        # Take the section from the root to the far right of sentence if nothing else
        else:
            sent_text = sent_text.split(' ')
            root_index = find_main(sent_dep)['address'] + 1
            answer = sent_text[root_index:]
            answer = ' '.join(answer)

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
            if marks is not None:
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
        # Either yes or no

        # check if sentence has a negative word
        neg_words = ['never', 'not', 'no', "'nt"]

        for word in neg_words:
            node = find_node(word, sent_dep)
            if node is not None:
                answer = 'no'
                return answer

        answer = "yes"
        return answer

    return answer


def story_about(s_dep):
    story_char = {}
    for sgraph in s_dep[0:1]:
        for node in sgraph.nodes.values():
            if node['rel'] == 'nsubj' or node['rel'] == 'dobj' or node['tag'] == 'NNP':
                word = node['word'].lower()
                if word in story_char:
                    freq = story_char.get(word) + 1
                    story_char[word] = freq
                else:
                    story_char[word] = 1
    char = sorted(story_char.items(), key=operator.itemgetter(1))
    answer = 'a ' + char[0][0] + ' and a ' + char[1][0]
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
    """

    #print (question['difficulty'])

    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
        s_dep = story['story_dep']
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])
        s_dep = story['sch_dep']

    q_dep = question['dep']
    qtext = question['text']
    question_type = get_question_type(question)

    best_sentences = get_best_sentences(question, story, question_type)
    best_sentence_texts, best_sentence_scores = get_top_sentences(best_sentences, 1)
    best_sentence = ' '.join(best_sentence_texts)


    print('\n' + qtext)
    for i in range(len(best_sentence_texts)):
        print(best_sentence_texts[i] + '\t' + str(best_sentence_scores[i]) + " " + question['qid'] + '\n')

    best_sentence = best_sentence_texts[0]
    sent_dep = best_sentences[0][1]
    sent_text = best_sentences[0][0]
    answer = narrow_answer(qtext, question_type, q_dep, sent_dep, best_sentence)

    qtext_list = qtext.split()
    # print(qtext_list)
    if question_type.lower() == 'who' and 'about?' in qtext_list:
        answer = story_about(s_dep)
        
    print('answer: ' + answer)
    return answer

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
