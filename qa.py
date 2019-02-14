
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk


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

    for x in question:
        dep = question["dep"]
        par = question["par"]
        text = question["text"]
        sid = question["sid"]
        difficulty = question["difficulty"]
        type = question["type"]
        qid = question["qid"]

        story_dep = story["story_dep"]
        sch_dep = story["sch_dep"]
        sch_par = story ["sch_par"]
        story_par = story["story_par"]
        sch = story["sch"]
        story_text = story["text"]
        sid = story["sid"]

        if type == "Story":
            answer = process_difficulty(dep, par, text, sid, difficulty, qid, story_text)
        else: #sch | (story | sch)
            answer = process_difficulty(dep, par, text, sid, difficulty, qid, sch)

    """"""
    answer = "whatever you think the answer is"

    ###     End of Your Code         ###
    return answer

def process_difficulty(dep, par, text, sid, difficulty, qid, story_text):
    if difficulty == "Easy":
        sentences = get_best_sentences(dep, par, text, sid, qid, story_text)

def get_best_sentences(dep, par, text, sid, qid, story):

    print("Dep")
    print(dep)
    print("Par")
    print(par)
    print("Text")
    print(text)

    best_sentences = []
    sentences = nltk.sent_tokenize(story)
    question_words = nltk.word_tokenize(text)

    for x in sentences:
        count = 0
        sentence_words = nltk.word_tokenize(x)
        for y in sentence_words:
            for k in question_words:
                if y == k:
                    count = count + 1
        if count > 3:
            best_sentences.append(x)

    print("Best Sentences")
    print(best_sentences)

    #sentence tokenize

    #Function to break apart question

    #get_sentence question
    #get sentence subject
    #get_sentence verb

    #Look into parse tree


    return sentences



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
