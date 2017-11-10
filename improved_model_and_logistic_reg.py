import json
import os
from pprint import pprint
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import math
import operator
from pycorenlp import StanfordCoreNLP
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tree import Tree
from collections import *
from gensim.models import word2vec
import gensim, logging
import tensorflow
from nltk.corpus import wordnet
from inflector import Inflector
import re

nlp = StanfordCoreNLP('http://localhost:9000')

###############################################################################
### a improved model of our baseline only using pos and ner
###############################################################################

### calculate cosine similarity between two sentences
def calculateSimilarity(a, b):
    vect = TfidfVectorizer(min_df=1)
    context_sentences = [a, b]
    tfidf = vect.fit_transform(context_sentences)
    prod = (tfidf * tfidf.T).A
    return prod[0][1]

### filter out non-content words
def remove_stop_words(sentence):
    result = ''
    for t in sentence.split():
        if t.lower() not in stop_words:
            result += (t + ' ')
    return result.rstrip()

### get all the content words in a sentence
def get_content_token_set(sentence):
    token_set = set()
    for pt in sentence.translate(translator).split():
        if pt.lower() not in stop_words:
            token_set.add(pt.lower())
    return token_set

### check the number of overlaps from context sentence tokens with question sentence tokens
def compare(q_text_set, c_text_set):
    score = 0
    for c_token in c_text_set:
        if c_token in q_text_set:
            score += 1
    return score

### find the sentence from context corpus which has the most overlaps with the question
def find_candidate_sentence_index(question, context_parsed):
    clist = list(context_parsed)
    clist.append(question)
    vect = TfidfVectorizer(min_df=1)
    #print(clist)
    tfidf = vect.fit_transform(clist)
    comp_mat = (tfidf * tfidf.T).A
    comp_vec = comp_mat[comp_mat.shape[0] - 1][0 : comp_mat.shape[0] - 1]
    #print(comp_mat)
    #print(comp_vec)
    return np.argmax(comp_vec)

### define question type
def find_question_type(question):
    for q in question.split():
        if q.lower() == 'what' or q.lower() == 'which':
            return 'NN'
        elif q.lower() == 'who' or q.lower() == 'where':
            return 'NNP'

### retrieve the question word from given question
def get_question_word(q_annotate):
    q_word = ''
    q_parsed = q_annotate['sentences'][0]['parse']
    for i in Tree.fromstring(q_parsed).subtrees():
        if i.label() == 'WP' or i.label() == 'WDT' or i.label() == 'WRB':
            q_word = ''.join(i.leaves())
    return q_word

### for each type of question, return the most possible ner's for the answer part
def get_question_type(q_word, question):
    q_word = q_word.lower()
    question = question.lower()
    inf = Inflector()
    question = inf.singularize(question)

    if q_word == 'what' or q_word == 'which':
        if 'what country' in question or \
            'what state' in question or \
            'what continental' in question or \
            'what place' in question or \
            'what city' in question or \
            'what province' in question or \
            'what river' in question or \
            'what region' in question or \
            'what area' in question or \
            'what nationality' in question or \
            'what town' in question or \
            'what borough' in question or \
            'what location' in question:
            return set(['LOCATION'])

        if 'what year' in question or \
            'what month' in question or \
            'what day' in question or \
            'what date' in question:
            return set(['DATE'])

        if 'what percentage' in question or 'what percent' in question:
            return set(['PERCENT'])

        if 'what company' in question or \
            'what group' in question or \
            'what organization' in question or \
            'what university' in question or \
            'what school' in question or \
            'what team' in question or \
            'what program' in question or \
            'what party' in question:
            return set(['ORGANIZATION'])

        if 'what artist' in question or \
            'what actor' in question or \
            'what actress' in question or \
            'what doctor' in question or \
            'what president' in question or \
            'what person' in question:
            return set(['PERSON'])

        if 'which country' in question or \
            'which state' in question or \
            'which continental' in question or \
            'which place' in question or \
            'which city' in question or \
            'which province' in question or \
            'which river' in question or \
            'which region' in question or \
            'which area' in question or \
            'which nationality' in question or \
            'which town' in question or \
            'which borough' in question or \
            'which location' in question:
            return set(['LOCATION'])

        if 'which year' in question or \
            'which month' in question or \
            'which day' in question or \
            'which date' in question:
            return set(['DATE'])

        if 'which percentage' in question or 'which percent' in question:
            return set(['PERCENT'])

        if 'which company' in question or \
            'which group' in question or \
            'which organization' in question or \
            'which university' in question or \
            'which school' in question or \
            'which team' in question or \
            'which program' in question or \
            'which party' in question:
            return set(['ORGANIZATION'])

        if 'which artist' in question or \
            'which actor' in question or \
            'which actress' in question or \
            'which doctor' in question or \
            'which president' in question or \
            'which person' in question:
            return set(['PERSON'])
    elif q_word == 'how':
        if 'how much' in question:
            return set(['MONEY', 'NUMBER'])
        if 'how long' in question or 'how old' in question:
            return set(['TIME', 'DURATION'])
        if 'how many' in question or 'how far' in question:
            return set(['NUMBER'])
    elif q_word == 'where':
        return set(['LOCATION','ORGANIZATION'])
    elif q_word == 'when':
        return set(['DATE', 'TIME', 'DURATION'])
    elif q_word == 'who':
        return set(['PERSON'])

    return set(['O'])

### find the most possible phrase as our answer for question given
def find_answer_from_candidate(question, q_annotate, candidate_sentence, q_word):
    cs_annotate = nlp.annotate(candidate_sentence, properties={
        'annotators': 'pos,ner,parse',
        'outputFormat': 'json'
        })

    if len(cs_annotate['sentences']) == 0:
        return ''

    q_np_set = set()
    q_np_filtered = ''
    q_np_vect = TfidfVectorizer(min_df=1)

    q_parsed = q_annotate['sentences'][0]['parse']
    for i in Tree.fromstring(q_parsed).subtrees():
        if i.label() == 'NP':
            np = ' '.join(i.leaves())
            q_np_filtered += (' ' + np)
            for p in i.leaves():
                q_np_set.add(p)

    q_np_rep = q_np_filtered.strip()

    cs_np_list = []
    cs_parsed = cs_annotate['sentences'][0]['parse']
    for i in Tree.fromstring(cs_parsed).subtrees():
        if i.label() == 'NP':
            np = ' '.join(i.leaves())
            cs_np_list.append(np)

    q_np_non_rep = ' '.join(q_np_set)

    sim_vect_rep = {}

    for cs_np in cs_np_list:
        vec = [q_np_rep, cs_np]
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(vec)
        comp_mat = (tfidf * tfidf.T).A
        sim_vect_rep[cs_np] = comp_mat[0][1]

    if len(sim_vect_rep) == 0:
        return ''

    min_sim = min([sim_vect_rep[k] for k in sim_vect_rep])
    cand_answer_list = [d for d in sim_vect_rep if sim_vect_rep[d] == min_sim]

    #print(cand_answer_list)
    #print(q_word)

    cand_ans_score = {}

    for ca in cand_answer_list:
        q_temp = str(question)
        q_temp = q_temp.replace(q_word, ca)
        #print(q_temp)
        vect = TfidfVectorizer(min_df=1)
        vec = [q_temp, candidate_sentence]
        tfidf = vect.fit_transform(vec)
        comp_mat = (tfidf * tfidf.T).A
        cand_ans_score[ca] = comp_mat[0][1]

    #print(cand_ans_score)
    return max(cand_ans_score, key=cand_ans_score.get)

### retrieve the most possible answer phrases from context based on the type of question
def find_answer_from_context_ner(question, question_annotate, context_annotate, q_word, qtype):
    possible_candidate_sentence_indices = {}
    possible_answers = {}
    possible_tokens = set()
    q_word = ''
    context_sentences = []
    for i in range(len(context_annotate['sentences'])):
        cas = context_annotate['sentences'][i]
        context_sentences.append(' '.join([t['word'] for t in cas['tokens']]))
        for t in cas['tokens']:
            if t['ner'] in qtype:
                possible_tokens.add(t['word'])
                if i in possible_candidate_sentence_indices:
                    possible_candidate_sentence_indices[i] += 1
                else:
                    possible_candidate_sentence_indices[i] = 1

        for t in Tree.fromstring(cas['parse']).subtrees():
            if t.label() == 'NP':
                np = ' '.join(t.leaves())

                for l in t.leaves():
                    if l in possible_tokens:
                        np_list = [n.strip() for n in re.split('; |, |\*|\n| and| or', np)]
                        if i in possible_answers:
                            possible_answers[i] += np_list
                        else:
                            possible_answers[i] = np_list
                        break
                    else:
                        break

    for i in possible_answers:
        possible_answers[i] = set(possible_answers[i])

    candidate_answer_score = {}
    for i in possible_candidate_sentence_indices:
        cand_sentence = ' '.join([d.translate(translator).strip() for d in context_sentences[i].split()])
        #print(cand_sentence)
        if i in possible_answers:
            for ca in possible_answers[i]:
                q_temp = str(question)
                q_temp = q_temp.replace(q_word, ca).translate(translator).strip()
                if ca not in candidate_answer_score:
                    candidate_answer_score[ca] = calculateSimilarity(q_temp, cand_sentence)
                else:
                    candidate_answer_score[ca] = max(candidate_answer_score[ca], calculateSimilarity(q_temp, cand_sentence))

    #print(candidate_answer_score)
    if len(candidate_answer_score) == 0:
        return ''
    return max(candidate_answer_score, key=candidate_answer_score.get)

### find the answer of question based on given context
def find_answer(question, context, context_annotate):
    context_sentences = [d.strip() for d in context.split('.')]
    context_sentences_filtered = [d.translate(translator).strip().lower() for d in context_sentences]
    context_sentences_filtered = [d for d in context_sentences_filtered if len(d) > 0]
    cs_index = find_candidate_sentence_index(question.translate(translator).strip().lower(), context_sentences_filtered)
    candidate_sentence = context_sentences[cs_index]
    question = question.translate(translator).strip()
    q_annotate = nlp.annotate(question, properties={
        'annotators': 'pos,ner,parse',
        'outputFormat': 'json'
        })
    q_word = get_question_word(q_annotate)
    answer = find_answer_from_candidate(question, q_annotate, candidate_sentence, q_word)
    #return answer
    answer_ner = None

    q_type = get_question_type(q_word, question)
    if 'O' not in q_type:
        answer_ner = find_answer_from_context_ner(question, q_annotate, context_annotate, q_word, q_type)
    #print(question)
    #print(q_type)
    #print(q_token_set)
    #print(candidate_sentence)
    #print(c_tag)
    #print(answer)
    #print(answer_ner)
    if answer_ner is not None:
        if len(answer_ner) > 0:
            answer = answer_ner
    return answer

def generate_answer_for_file(file_name):
    with open(file_name) as data_file:
        data = json.load(data_file)
    our_pred = {}
    for i in range(0, len(data['data'])):
    #for i in range(0, 1):
        dataset = data['data'][i]
        for j in range(len(dataset['paragraphs'])):
        #for j in range(0, 1):
            question_bucket = dataset['paragraphs'][j]
            context = question_bucket['context']

            context_annotate = nlp.annotate(context, properties={
                'annotators': 'pos,ner,parse',
                'outputFormat': 'json'
            })

            for k in range(len(question_bucket['qas'])):
                question = question_bucket['qas'][k]['question']
                qid = question_bucket['qas'][k]['id']
                #answer = find_answer(question, context, context_annotate)
                #our_pred[qid] = answer
                try:
                    answer = find_answer(question, context, context_annotate)
                    our_pred[qid] = answer
                except:
                    our_pred[qid] = ''
                    pass
    return our_pred

### Generate answer files for development set and test sets
our_pred = generate_answer_for_file('development.json')
with open('pred_development.json', 'w') as outfile:
    json.dump(our_pred, outfile)


our_pred = generate_answer_for_file('testing.json')
with open('pred_testing.json', 'w') as outfile:
    json.dump(our_pred, outfile)

###############################################################################
### logistic regression approach
###############################################################################

with open('training.json') as data_file:
    train_data = json.load(data_file)

with open('development.json') as data_file:
    dev_data = json.load(data_file)

with open('testing.json') as data_file:
    test_data = json.load(data_file)

### read all contexts, questions and answers from
def getCorpus(data):
    contexts = []
    questions = []
    answers = []
    for i in range(0, len(data['data'])):
    #for i in range(0, 1):
        dataset = data['data'][i]
        for j in range(len(dataset['paragraphs'])):
        #for j in range(0, 1):
            question_bucket = dataset['paragraphs'][j]
            contexts.append(question_bucket['context'])
            for k in range(len(question_bucket['qas'])):
                questions.append(question_bucket['qas'][k]['question'])
                answers.append(question_bucket['qas'][k]['answers'][0]['text'])
    return contexts, questions, answers

train_contexts, train_questions, train_answers = getCorpus(train_data)
dev_contexts, dev_questions, dev_answers = getCorpus(dev_data)
test_contexts, test_questions, test_answers = getCorpus(test_data)

translator = str.maketrans('', '', string.punctuation)

def getVectorizedCorpusModel(contexts, questions, answers):
    contexts_filtered = [c.split('.') for c in contexts]
    contexts_out = [d.translate(translator).split() for c in contexts_filtered for d in c]
    questions_out = [c.translate(translator).split() for c in questions]
    answers_out = [c.translate(translator).split() for c in answers]
    out = contexts_out + questions_out + answers_out
    return gensim.models.Word2Vec(out, size=200, window=5, min_count=1, workers=4)

model = getVectorizedCorpusModel(train_contexts + dev_contexts + test_contexts,
                                 train_questions + dev_questions + test_questions,
                                 train_answers + test_questions)


### filter out non-content words
def remove_stop_words(sentence):
    result = ''
    for t in sentence.split():
        if t.lower() not in stop_words:
            result += (t + ' ')
    return result.rstrip()

### Get the list of pos and ner tags for a sentence
def getPosNerList(sentence):
    output = nlp.annotate(sentence, properties={
      'annotators': 'pos,ner',
      'outputFormat': 'json'
      })
    posList = []
    nerList = []
    for s in output['sentences']:
        for t in s['tokens']:
            posList.append(t['pos'])
            nerList.append(t['ner'])
    return posList,nerList

### get the fraction of each ner in a sentence
def getNerFracVector(annotateList):
    nerType = {'PERSON': 0, 'LOCATION': 1, 'ORGANIZATION': 2, 'DATE': 3, 'TIME': 4, 'MONEY': 5, 'PERCENT': 6, 'MISC': 7}
    freq = [0] * len(nerType)
    for a in annotateList:
        if a in nerType:
            freq[nerType[a]] += 1

    if len(annotateList) > 0:
        for i in range(len(freq)):
            freq[i] /= (1.0 * len(annotateList))
    return freq

### get the fraction of each noun phrase pos in a sentence
def getPosFracVector(annotateList):
    posType = {'NNP': 0, 'NNPS': 1, 'NN': 2, 'NNS': 3}
    freq = [0] * len(posType)
    for a in annotateList:
        if a in posType:
            freq[posType[a]] += 1

    if len(annotateList) > 0:
        for i in range(len(freq)):
            freq[i] /= (1.0 * len(annotateList))
    return freq

### retrieve the question word and indicate its occurance in a vector
def getQuestionTypeVector(question):
    qType = {'what': 0, 'where': 1, 'which': 2, 'who': 3, 'when': 4, 'why': 5, 'how': 6}
    qWord = 'what'
    for d in question:
        if d.lower() in qType:
            qWord = d.lower()
    vec = [0] * 7
    vec[qType[qWord]] = 1
    return vec

### Build our feature matrix based on question and context
def getFeatureForQuestionAndContext(question, q_pos, q_ner, context, c_pos, c_ner):
    q = question.translate(translator).split()
    c = context.translate(translator).split()

    feat = sum(model[k] for k in q) / len(q)
    if len(c) > 0:
        feat += sum(model[k] for k in c) / len(c)

    # Use word embeddings for current question
    #feat = (sum(model[k] for k in q) / len(q) + sum(model[k] for k in c) / len(c)).tolist()
    feat = feat.tolist()

    # Add similarity bewteen question and current sentence
    feat.append(calculateSimilarity(question, context))

    # Add question Type
    feat += getQuestionTypeVector(q)

    q_pos_frac = getPosFracVector(q_pos)
    c_pos_frac = getPosFracVector(c_pos)

    # Add the interaction between the overlap of pos tags in question and context
    posOverlap = [0] * len(q_pos_frac)
    for i in range(len(posOverlap)):
        posOverlap[i] = q_pos_frac[i] * c_pos_frac[i]

    feat += posOverlap

    # Add the interaction between the overlap of ner tags in question and context
    q_ner_frac = getNerFracVector(q_ner)
    c_ner_frac = getNerFracVector(c_ner)
    nerOverlap = [0] * len(q_ner_frac)
    for i in range(len(nerOverlap)):
        nerOverlap[i] = q_ner_frac[i] * c_ner_frac[i]

    feat += nerOverlap

    return feat

### get feature matrix and label vector for a paragraph in our json file
def getFeatureAndLabelForParagraph(paragraph):
    contexts = [d.strip() for d in paragraph['context'].split('.') if len(d) > 0]
    X = []
    y = []
    c_pos = []
    c_ner = []
    for i in range(len(contexts)):
        c_pos_i, c_ner_i = getPosNerList(remove_stop_words(contexts[i]))
        c_pos.append(c_pos_i)
        c_ner.append(c_ner_i)

    for qas in paragraph['qas']:
        question = qas['question']
        answer = qas['answers'][0]['text']
        q_pos, q_ner = getPosNerList(remove_stop_words(question))

        for i in range(len(contexts)):
            if len(contexts[i]) > 0 and len(question) > 0:
                X.append(getFeatureForQuestionAndContext(question, q_pos, q_ner, contexts[i], c_pos[i], c_ner[i]))
                y.append(1 if answer in contexts[i] else 0)
    return X, y, len(contexts)

### get feature matrix and label vector for entire data in a json file
### this function is only for training data
def getFeatureAndLabel(data):
    X = []
    y = []
    counter = 0

    for i in range(0, len(data['data'])):
        if counter > 3000:
            break
    #for i in range(0, 1):
        dataset = data['data'][i]
        for j in range(len(dataset['paragraphs'])):
        #for j in range(0, 1):
            counter += 1
            if counter % 500 == 0:
                print(counter)
            question_bucket = dataset['paragraphs'][j]
            Xj, yj, _ = getFeatureAndLabelForParagraph(question_bucket)
            X += Xj
            y += yj
    return X, y

### get feature matrix for a context and its question
### this function is only for development or testing data
def getFeatureForContextAndQuestion(contexts, question):
    X = []
    c_pos = []
    c_ner = []
    for i in range(len(contexts)):
        c_pos_i, c_ner_i = getPosNerList(remove_stop_words(contexts[i]))
        c_pos.append(c_pos_i)
        c_ner.append(c_ner_i)

    q_pos, q_ner = getPosNerList(remove_stop_words(question))

    for i in range(len(contexts)):
        if len(contexts[i]) > 0 and len(question) > 0:
            X.append(getFeatureForQuestionAndContext(question, q_pos, q_ner, contexts[i], c_pos[i], c_ner[i]))
    return X

### get the index of vector where the value is 1
def get_cand_index(y):
    for i in range(len(y)):
        if y[i] == 1:
            return i
    return 0

### generate answer for input question file using the logistic regression model
def generate_answer_for_file_logistic_model(file_name, model):
    with open(file_name) as data_file:
        data = json.load(data_file)
    our_pred = {}
    for i in range(0, len(data['data'])):
    #for i in range(0, 1):
        dataset = data['data'][i]
        for j in range(len(dataset['paragraphs'])):
        #for j in range(0, 1):
            question_bucket = dataset['paragraphs'][j]
            context = question_bucket['context']
            contexts = [d.strip() for d in context.split('.') if len(d) > 0]
            context_annotate = nlp.annotate(context, properties={
                'annotators': 'pos,ner,parse',
                'outputFormat': 'json'
            })
            for k in range(len(question_bucket['qas'])):
                question = question_bucket['qas'][k]['question']
                qid = question_bucket['qas'][k]['id']
                X = getFeatureForContextAndQuestion(contexts, question)
                y = clf.predict(X)
                num_ans = sum(y)
                answer = ''
                ### If we can be certain that one of the sentences from context
                ### is the answer, we use this as our candidate sentence.
                ### Otherwise we use the normal way to find our answer
                if num_ans == 1:
                    cand_i = get_cand_index(y)
                    candidate_sentence = contexts[cand_i]
                    q_annotate = nlp.annotate(question, properties={
                        'annotators': 'pos,ner,parse',
                        'outputFormat': 'json'
                    })
                    try:
                        answer = find_answer_from_candidate(question, q_annotate, candidate_sentence, get_question_word(q_annotate))
                    except:
                        pass
                else:
                    try:
                        answer = find_answer(question, context, context_annotate)
                    except:
                        pass

                our_pred[qid] = answer

    return our_pred

### training of our logistic regression, using the first 3000 paragraphs
X, y = getFeatureAndLabel(train_data)
X = np.array(X)
y = np.array(y)

#clf = linear_model.SGDClassifier()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
clf.fit(X, y)

### Generate answer files for development set and test sets
our_pred = generate_answer_for_file_logistic_model('development.json', model)
with open('pred_development_log.json', 'w') as outfile:
    json.dump(our_pred, outfile)


our_pred = generate_answer_for_file_logistic_model('testing.json', model)
with open('pred_testing_log.json', 'w') as outfile:
    json.dump(our_pred, outfile)
