### CS 5740 Implementation of Baseline System

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

q_type = {'what':'NN', 'who':'NNP'}
stop_words = set(stopwords.words('english'))

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
def find_candidate_sentence(q_token_set, context):
    candidate = context[0]
    score = compare(q_token_set, get_content_token_set(candidate))
    for i in range(1, len(context)):
        temp_candi = context[i]
        temp_score = compare(q_token_set, get_content_token_set(temp_candi))
        if temp_score >= score:
            score = temp_score
            candidate = temp_candi
    return candidate

### define question type
def find_question_type(question):
    for q in question.split():
        if q.lower() == 'what' or q.lower() == 'which':
            return 'NN'
        elif q.lower() == 'who' or q.lower() == 'where':
            return 'NNP'

### find the possible answer from candidate sentence based on calculating
### the distance from
def find_possible_answer_with_distance(q_token_set, candidate_sentence, q_type):
    token_positions = {}
    cand_tokens = candidate_sentence.split()
    for i in range(len(cand_tokens)):
        if cand_tokens[i].lower() in q_token_set:
            token_positions[cand_tokens[i]] = i
    #print(token_positions)
    c_tag = nltk.pos_tag(word_tokenize(candidate_sentence))
    possible_answers = set()
    for c in c_tag:
        if c[1] == q_type and c[0].lower() not in q_token_set:
            possible_answers.add(c[0])

    answer_possitions = {}
    for i in range(len(cand_tokens)):
        if cand_tokens[i] in possible_answers:
            answer_possitions[cand_tokens[i]] = i
    answer_score = {}
    for ap in answer_possitions:
        position = answer_possitions[ap]
        score = 0
        for tp in token_positions:
            score += abs(position - token_positions[tp])
        answer_score[ap] = score
    answer_score = sorted(answer_score.items(), key=operator.itemgetter(1))
    #print(possible_answers)
    #print(answer_possitions)
    #print(answer_score)

    answer = str()
    for ans in answer_score:
        answer += (ans[0] + ' ')

    return answer.rstrip()

### find the answer to certain question from certain context
def find_answer(question, context):
    q_tag = nltk.pos_tag(word_tokenize(question))
    q_token_set = get_content_token_set(question)
    q_type = find_question_type(question)
    context_sentences = [c.lstrip().translate(translator) for c in context.split('.') if len(c) > 0]
    candidate_sentence = find_candidate_sentence(q_token_set, context_sentences)
    #c_tag = nltk.pos_tag(word_tokenize(candidate_sentence))
    answer = find_possible_answer_with_distance(q_token_set, candidate_sentence, q_type)
    #print(question)
    #print(q_type)
    #print(q_token_set)
    #print(candidate_sentence)
    #print(c_tag)
    #print(answer)
    return answer

### Generate json model with correct format
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
            for k in range(len(question_bucket['qas'])):
                question = question_bucket['qas'][k]['question']
                qid = question_bucket['qas'][k]['id']
                answer = find_answer(question, context)
                our_pred[qid] = answer
    return our_pred

### Generate answer files for development set and test sets
our_pred = generate_answer_for_file('development.json')
with open('pred_development_baseline.json', 'w') as outfile:
    json.dump(our_pred, outfile)


our_pred = generate_answer_for_file('testing.json')
with open('pred_testing_baseline.json', 'w') as outfile:
    json.dump(our_pred, outfile)
