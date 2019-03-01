#!/usr/bin/env python

import re, sys, nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer

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


def find_answer(qgraph, sgraph, rel):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    print(qword)

    snode = find_node(qword, sgraph)
    print(snode)

    if snode is None:
        for node in qgraph.nodes.values():
            if node['rel'] == 'nsubj' or node['rel'] == 'nmod':
                qword = node['word']
                snode = find_node(qword, sgraph)
                snode = sgraph.nodes[snode.get('head', None)]
                if snode is None:
                    smain = find_main(sgraph)
                    snode = smain["word"]
                print(snode)
            else:
                 return None

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

        # if we can't find dependents from parent, then look at case
        else:   
            print("looking for case")   
            for node in sgraph.nodes.values():
                if node['rel'] == 'case':
                    print(node)
                    for item in node["deps"]:
                        if item == rel:
                            # print(rel)
                            address = node["deps"][item][0]
                            rnode = sgraph.nodes[address]
                            deps.append(rnode)
                            return " ".join(dep["word"] for dep in deps)

    return None


if __name__ == '__main__':
    driver = QABase()

    # Get the first question and its story
    q = driver.get_question("mc500.train.23.15")
    story = driver.get_story(q["sid"])
    # get the dependency graph of the first question
    qgraph = q["dep"]
    #print("qgraph:", qgraph)
    qtext = q['text']

    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    sgraph = story["story_dep"][17]
    print(sgraph)
    stext = story['text']
    print(stext)
    print(q['type'])

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
    
    print(qtext)
    answer = find_answer(qgraph, sgraph, "nsubj")
    if not answer:
        answer = find_answer(qgraph, sgraph, "nmod")
    if not answer:
        answer = find_answer(qgraph, sgraph, "amod")
    if not answer:
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
    print("answer:", answer)

