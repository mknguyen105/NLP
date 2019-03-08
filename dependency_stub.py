#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from qa_engine.base import QABase
  
    
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root' or node['rel'] == 'Root':
            return node
    return None


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
                print(nword)
                return node
    return None


def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
        
    return results


# finds lemma/hyponym/hypernym of word in sentence and returns its node
def find_h_nyms(word, graph):
    synsets = wn.synsets(word)
    for synset in synsets:
        lemmas = synset.lemma_names()
        for lemma in lemmas:
            node = find_node(lemma, graph)
            if node is not None:
                print('lemma: ' + lemma)
                return node
        hyponyms = synset.hyponyms()
        for hypo in hyponyms:
            hypo = hypo.name()[0:hypo.name().index(".")]
            node = find_node(hypo, graph)
            if node is not None:
                print('hypo: ' + hypo)
                return node
        hypernyms = synset.hypernyms()
        for hyper in hypernyms:
            hyper = hyper.name()[0:hyper.name().index(".")]
            node = find_node(hyper, graph)
            if node is not None:
                print('hyper: ' + hyper)
                return node
    return None


# finds all hyponyms/hypernyms of words in list
def find_all_h_nyms(wlist):
    hlist = []
    for word in wlist:
        synsets = wn.synsets(word)
        for synset in synsets:
            hyponyms = synset.hyponyms()
            for hypo in hyponyms:
                hlist.append(hypo.name()[0:hypo.name().index(".")])
            hypernyms = synset.hypernyms()
            for hyper in hypernyms:
                hlist.append(hyper.name()[0:hyper.name().index(".")])
    return hlist


def find_answer(qgraph, sgraph, rel):
    qmain = find_main(qgraph)
    qword = qmain["word"]

    qtext = []
    for node in qgraph.nodes.values():
        if node['word'] is not None:
            qtext.append(node["word"])
    
    # if qword is a question type
    if qword.lower() in ['who', 'what', 'when', 'where', 'why', 'how', 'which']:
        for node in qgraph.nodes.values():
            if node['rel'] == 'nsubj' or node['rel'] == 'nmod' or node['rel'] == 'nmod':
                qmain = node
                qword = qmain["word"]

    print(qword)

    snode = find_node(qword, sgraph)

    # if qword is not found in sentence
    # find hyponym/hypernym of qword in sentence
    if snode is None:
        snode = find_h_nyms(qword, sgraph)

    # if qword is still not found in sentence
    # find nsubj/nmod/dobj from question in sentence
    if snode is None:
        # print(qgraph)
        for node in qgraph.nodes.values():
            # print(node)
            if node['rel'] == 'nsubj' or node['rel'] == 'nmod' or node['rel'] == 'dobj' and node['lemma'] not in ['who', 'what', 'when', 'where', 'why', 'how', 'which']:
                qword = node['word']
                snode = find_node(qword, sgraph)
                if snode is None:
                    snode = find_h_nyms(qword, sgraph)
                # if snode is found, exit for loop
                if snode is not None:
                    break
                    # snode = sgraph.nodes[snode.get('head', None)]
        if snode is None:
            snode = find_main(sgraph)
    
    print(snode)

    deps = []

    for node in sgraph.nodes.values():
        #print("node[head]=", node["head"])
        if node.get('head', None) == snode["address"]:
            if node['rel'] == rel:
                #print(node["word"], node["rel"])
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)

    # if we can't find dependents from main verb, then look at parent dependent
    if len(deps) == 0:    
        parent_node = sgraph.nodes[snode.get('head', None)]
        print(parent_node)
        if parent_node['word'] is not None:
            for node in sgraph.nodes.values():
                if node.get('head', None) == parent_node["address"]:
                    if node['rel'] == rel:
                        deps = get_dependents(node, sgraph)
                        deps = sorted(deps+[node], key=operator.itemgetter("address"))
                        return " ".join(dep["word"] for dep in deps)
            if parent_node['rel'] == rel:
                deps = get_dependents(parent_node, sgraph)
                deps = sorted(deps+[parent_node], key=operator.itemgetter("address"))
                # print(" ".join(dep["word"] for dep in deps))
                # remove question word in answer
                for dep in deps:
                    if dep == snode:
                        sdep = get_dependents(snode, sgraph)
                        sdep.append(dep)
                        print(" ".join(s['word'] for s in sdep))
                        adeps = []
                        for d in deps:
                            if d not in sdep:
                                adeps.append(d)
                return " ".join(dep["word"] for dep in adeps)

        # if we can't find dependents from parent, then look at case
        else:   
            print("looking for case")   
            for node in sgraph.nodes.values():
                if node['rel'] == 'case':
                    print(node)
                    # add case if it's not in question
                    if node['word'] not in qtext:
                        deps.append(node)
                    for item in node["deps"]:
                        if item == rel:
                            # print(rel)
                            address = node["deps"][item][0]
                            rnode = sgraph.nodes[address]
                            deps.append(rnode)
                            return " ".join(dep["word"] for dep in deps)

    return None


def remove_case(cdeps):
    deps = []
    for node in cdeps:
        if node['rel'] != 'case':
            deps.append(node)
    return deps


def find_who_answer(qtext, qgraph, sgraph):
    qmain = find_main(qgraph)
    qword = qmain["word"]

    # loophole for answers with narrator and young man
    # remove young man from qtext, so we can get it as an answer
    if 'narrator' in qtext or 'young man' in qtext:
        qtext = re.sub('narrator', '', qtext)
        qtext = re.sub('young man', '', qtext)
        # print(qtext)

    qwords = nltk.word_tokenize(qtext)
    # make sure that hyponyms/hypernyms from question not in answer
    qwords_h = find_all_h_nyms(qwords)
    
    # if qword is a question type
    if qword.lower() in ['who', 'what', 'when', 'where', 'why', 'how', 'which']:
        for node in qgraph.nodes.values():
            if node['rel'] == 'nsubj' or node['rel'] == 'nmod' or node['rel'] == 'dobj':
                qmain = node
                qword = qmain["word"]

    print(qword)

    snode = find_node(qword, sgraph)

    # if qword is not found in sentence
    # find hyponym/hypernym of qword in sentence
    if snode is None:
        snode = find_h_nyms(qword, sgraph)

    # if qword is still not found in sentence
    # find nsubj/nmod/dobj from question in sentence
    if snode is None:
        # print(qgraph)
        for node in qgraph.nodes.values():
            # print(node)
            if node['rel'] == 'nsubj' or node['rel'] == 'nmod' or node['rel'] == 'dobj' and node['lemma'] not in ['who', 'what', 'when', 'where', 'why', 'how', 'which']:
                qword = node['word']
                snode = find_node(qword, sgraph)
                if snode is None:
                    snode = find_h_nyms(qword, sgraph)
                # if snode is found, exit for loop
                if snode is not None:
                    break
                    # snode = sgraph.nodes[snode.get('head', None)]
        if snode is None:
            snode = find_main(sgraph)
    
    print(snode)

    deps = []

    for node in sgraph.nodes.values():
        # print("node[head]=", node["head"])
        if node.get('head', None) == snode["address"]:
            # print(node["word"], node["rel"])
            if node['rel'] == 'nsubj' and node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                # deps = remove_case(deps)
                return " ".join(dep["word"] for dep in deps)
            elif node['rel'] == 'nmod' and node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)
            elif node['rel'] == 'dobj' and node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                deps = get_dependents(node, sgraph)
                deps = sorted(deps+[node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)

    # if we can't find dependents from main verb, then look at parent dependent
    if len(deps) == 0: 
        print('looking at parent')   
        parent_node = sgraph.nodes[snode.get('head', None)]
        print(parent_node)
        if parent_node['word'] is not None:
            for node in sgraph.nodes.values():
                if node.get('head', None) == parent_node["address"]:
                    # print(node["word"], node["rel"])
                    if node['rel'] == 'nsubj' and node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                        deps = get_dependents(node, sgraph)
                        deps = sorted(deps+[node], key=operator.itemgetter("address"))
                        return " ".join(dep["word"] for dep in deps)
                    elif node['rel'] == 'nmod' and node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                        deps = get_dependents(node, sgraph)
                        deps = sorted(deps+[node], key=operator.itemgetter("address"))
                        return " ".join(dep["word"] for dep in deps)
                    elif node['rel'] == 'dobj' and node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                        deps = get_dependents(node, sgraph)
                        deps = sorted(deps+[node], key=operator.itemgetter("address"))
                        return " ".join(dep["word"] for dep in deps)

            if parent_node['rel'] == 'nsubj' or parent_node['rel'] == 'nmod' or parent_node['rel'] == 'dobj' and parent_node['word'] not in qtext and not [w for w in qwords_h if node['word'] in w]:
                deps = get_dependents(parent_node, sgraph)
                deps = sorted(deps+[parent_node], key=operator.itemgetter("address"))
                return " ".join(dep["word"] for dep in deps)

        # if we can't find dependents from parent, then find any nsubj
        else:   
            print("looking for nsubj")
            deps = []   
            for node in sgraph.nodes.values():
                for item in node["deps"]:
                    if item == 'nsubj' or item == 'compound' or item == 'det' or item == 'amod':
                        address = node["deps"][item][0]
                        rnode = sgraph.nodes[address]
                        
                        # make sure we only get one nsubj
                        if rnode['word'] not in qtext:
                            if len(deps) > 1:
                                rel = []
                                for d in deps:
                                    rel.append(d['rel'])
                                if item not in rel:
                                    deps.append(rnode)
                            else:
                                deps.append(rnode)

            return " ".join(dep["word"] for dep in deps)

    if len(deps) == 0:
        print("looking for nsubj")
        deps = []   
        for node in sgraph.nodes.values():
            for item in node["deps"]:
                if item == 'nsubj' or item == 'compound' or item == 'det' or item == 'amod':
                    address = node["deps"][item][0]
                    rnode = sgraph.nodes[address]
                    
                    # make sure we only get one nsubj
                    if len(deps) > 1:
                        rel = []
                        for d in deps:
                            rel.append(d['rel'])
                        if item not in rel:
                            deps.append(rnode)
                    else:
                        deps.append(rnode)

        return " ".join(dep["word"] for dep in deps)

    return None


def last_effort_answer(sgraph):
    # if nothing works, then justt return the root of the sentence along with any compound, det, amod
    deps = []
    node = find_main(sgraph)
    deps.append(node)
    for item in node["deps"]:
        if item == 'compound' or item == 'det' or item == 'amod':
            address = node["deps"][item][0]
            rnode = sgraph.nodes[address]
            deps.append(rnode)
            deps = sorted(deps, key=operator.itemgetter("address"))
    answer = " ".join(dep["word"] for dep in deps)

    return answer


def last_effort_where_answer(sgraph):
    print("looking at all cases")  
    deps = [] 
    cases = []
    for node in sgraph.nodes.values():
        if node['rel'] == 'case':
            # print(node)
            cases.append(node)
    for node in cases:
        # add next few nodes until we reach a noun
        while node['tag'] is not None and 'NN' not in node['tag']:
            deps.append(node)
            if sgraph.nodes[node['address'] + 1] is not None:
                node = sgraph.nodes[node['address'] + 1]
            else:
                break
        if node['word'] is not None:
            deps.append(node)

    return " ".join(dep["word"] for dep in deps)


if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("mc500.train.23.23")
    story = driver.get_story(q["sid"])
    # get the dependency graph of the first question
    qgraph = q["dep"]
    #print("qgraph:", qgraph)
    qtext = q['text']

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    if q['type'] == 'story' or q['type'] == 'Story':
        stext = story['text']
        sgraph = story["story_dep"][6]
    else:
        stext = story['sch']
        sgraph = story["sch_dep"][15]
    
    # print(qgraph)
    print(sgraph)
    print(stext)
    # print(q['type'])

    '''
    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()
    '''
    
    print(qtext)

    # who
    answer = find_who_answer(qtext, qgraph, sgraph)

    # where
    '''
    answer = find_answer(qgraph, sgraph, "nmod")
    if not answer:
        answer = find_answer(qgraph, sgraph, "nmod:poss")
    if not answer:
        answer = find_answer(qgraph, sgraph, "dobj")
    if not answer:
        answer = last_effort_where_answer(sgraph)
    '''

    # when
    '''
    answer = find_answer(qgraph, sgraph, "nmod")
    if not answer:
        answer = find_answer(qgraph, sgraph, "nmod:tmod")
    if not answer:
        answer = find_answer(qgraph, sgraph, "advmod")
    if not answer:
        answer = last_effort_where_answer(sgraph)
    '''
    
    print("answer:", answer)

