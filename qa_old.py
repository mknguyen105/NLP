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
    story_subjects = find_subjects(story['story_dep'])
    qsubj = find_subjects(question['dep'])

    if question['type'] == "Story":
        sentences = get_sentences(story['text'])
    else:  # sch | (story | sch)
        sentences = get_sentences(story['sch'])


    question_text = question['text']
    question_words = nltk.word_tokenize(str(question_text))

    type = get_question_type(question)

    print('\n' + question_text + '\n')

    best_sentences = []

    # These are used for tuning in what and why questions. Verb_comp gets similar verbs in what questions, and to_verb
    # picks up to + VERB relations for why questions like "to eat" or "to visit"
    verb_comp = False
    to_verb = False

    #Classify question type
    verb_count = 0
    noun_count = 0
    prop_noun_count = 0
    keywords = []

    # Nouns, proper nouns, verbs and patterns and words specific to answers for each question type add differing points
    # to the sentences overlap score. Key count is how many points the key words provide.
    if type == "what":
        verb_count = 5
        noun_count = 5
        prop_noun_count = 2
        verb_comp = True
    elif type == "where":
        #Look for prepositional noun
        keywords = ["in", "on", "at", "behind", "below", "beside", "above", "across", "along", "below", "between", "under",
              "near", "inside", "from"]
        key_count = 5
        verb_count = 1
        noun_count = 1
    elif type == "who":
        #Look for person
        verb_count = 2
        noun_count = 2
        prop_noun_count = 3
    elif type == "why":
        key_count = 4
        keywords = ['because', 'so that', 'in order to']
    elif type == 'when':
        keywords = ['today', 'yesterday', "o'clock", 'year', 'month', 'hour', 'minute', 'second', 'week', 'after', 'before']
        key_count = 5


    for sent in sentences:
        count = 0
        for token, pos in sent:

            # Check if the token is a keyword and increase score if it is
            if token.lower() in keywords:
                count = count + key_count

            # Capture to + VERB relations to help with answers to why questions
            if to_verb:
                if pos == 'VBP' or pos == "VB":
                    count += 1
                to_verb = False
            if type == 'why' and token == 'to':
                to_verb = True


            for qword in question_words:

                # If the token is a verb, check if it relates strongly to any of the question words
                if (pos == "VBP" or pos == "VB" or pos == "VBD" or pos == "VBN") and verb_comp:
                   count += get_verb_similarity(qword, token)

                # Stem the question words and remove the stopwords before comparing them to the token
                if stemmer.stem(qword) == stemmer.stem(token) and token not in STOPWORDS:
                #if stemmer.stem(qword) == stemmer.stem(token):

                    # Add noun_count points for each noun verb, and verb_count points for each verb found
                    if pos == "VBP" or pos == "VB" or pos == "VBD" or pos == "VBN":
                        count = count + verb_count
                    elif pos == "NN" or pos == "NNS":
                        count = count + noun_count
                    elif pos == "NNP" or pos == "NNPS" or pos == "PRP":
                        count += prop_noun_count
                    else:
                        count = count + 1

        # Join sentences and their scores and print out for debugging
        joined_sentence = ' '.join([word for (word,tag) in sent])
        print(joined_sentence + " " + str(count))

        # Create a list of tuples containing (raw sentence, tokenized sentence with tags, overlap score)
        sent_tuple = (joined_sentence, sent, count)
        best_sentences.append(sent_tuple)

    # Sort list of tuples in desc order by their overlap score
    best_sentences.sort(key=lambda x: x[2], reverse=True)

    # Return the list of sentence tuples from above
    return best_sentences




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