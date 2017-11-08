import corenlp
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import numpy as np
import json
import queue
import random

nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = nltk.corpus.stopwords.words('english')
PUNCTUATIONS = [',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%']
QUESTIONTYPE = ['what','who','where','when','how','why','which','other']
TIMENER = ['DATE','DURATION','TIME']
NUMERICAL = ['MONEY','NUMBER','ORDINAL','PERCENT']
MAX_SIZE = 100
ANSWER_SET = defaultdict(str)
OUTPUTFILE = 'result1.json'
INPUTFILE = 'singleparagraph.json'
#paragraph: qas
#				q
#				id
#				a
#           context
#NNS, NNP, NNPS, NN
#NER: NAME: PERSON, LOCATION, ORGANIZATION, MISC
#NUMERICAL: MONEY, NUMBER, ORDINAL, PERCENT
#TEMPORAL: DATE, TIME, DURATION, SET
#NLP = StanfordCoreNLP('http://localhost:9000')

file = INPUTFILE
with open(file, 'r') as f:
	data = f.read()
line = json.loads(data)

class span:
	def __init__(self):
		self.tokens = []
		self.type = 'O'

#dfs
def getTokes_fromTree(tree):
	result = ""
	if len(tree.child) > 0:
		for child in tree.child:
			if child.value == "NP":
				return "", True
			t, h = getTokes_fromTree(child)
			if h:
				return "", True
			else:
				result += t + " "
	else:
		return tree.value, False
	return result, False

def getWHNP_fromTree(tree):
	result = ""
	if len(tree.child) > 0:
		for child in tree.child:
			if child.value == "WHNP":
				return "", True
			t, h = getTokes_fromTree(child)
			if h:
				return "", True
			else:
				result += t + " "
	else:
		return tree.value, False
	return result, False

#deal with the parse tree
#e.g. TEST_SENTENCE = "The writings of Samuel Pepys describe the pub as the heart of England."
#return [['The', 'writings'], ['the', 'pub'], ['Samuel', 'Pepys'], ['the', 'heart'], ['England']]
def parse_tree(Q):
	tokens_q = queue.Queue(maxsize=MAX_SIZE)
	while not Q.empty():
		tree = Q.get()
		if tree.value == 'NP':
			tokens_q.put(tree)
		if len(tree.child) > 0:
			for child in tree.child:
				Q.put(child)
	#get all tokens such as [tk1,tk2,tk3] from tree
	tokens_list = []
	while not tokens_q.empty():
		tr = tokens_q.get()
		(r, h) = getTokes_fromTree(tr)
		if not h:
			tl = r.split()
			tokens_list.append(tl)
	return tokens_list

def parse_tree_WHNP(Q):
	tokens_q = queue.Queue(maxsize=MAX_SIZE)
	while not Q.empty():
		tree = Q.get()
		if tree.value == 'WHNP':
			tokens_q.put(tree)
		if len(tree.child) > 0:
			for child in tree.child:
				Q.put(child)
	#get all tokens such as [tk1,tk2,tk3] from tree
	tokens_list = []
	while not tokens_q.empty():
		tr = tokens_q.get()
		(r, h) = getWHNP_fromTree(tr)
		if not h:
			tl = r.split()
			tokens_list.append(tl)
	return tokens_list

#l1: [['The', 'writings'], ['the', 'pub'], ['Samuel', 'Pepys'], ['the', 'heart'], ['England']]
#l2: token1, token2,……, tokenn
def generateSpans(l1, l2):
	span_list = []
	for s in l1:
		sp = span()
		for word in s:
			sp.tokens.append(word)
			for wordl2 in l2:
				w = wordl2.word
				if word == w:
					ner = wordl2.ner
					if ner != "O":
						sp.type = ner
					break;
		span_list.append(sp)
	return span_list

def find_which(which_sentence, replacedSpan, client):
	c = client.annotate(which_sentence)
	which_parse_tree = c.sentence[0].parseTree
	Q = queue.Queue(maxsize=MAX_SIZE)
	Q.put(which_parse_tree)
	tokens_list = parse_tree(Q)
	spanList = generateSpans(tokens_list, c.sentence[0].token)
	index = 0
	for j in range(len(spanList)):
		ts = " ".join(spanList[j].tokens)
		if 'which' in ts:
			index = j
			break;
	if spanList:
		a = spanList[index].tokens
		a1 = " ".join(a)
		b = replacedSpan.tokens
		b1 = " ".join(b)
		try:
			which_sentence = which_sentence.replace(a1, b1)
		except Exception as e:
			pass
	else:
		b = replacedSpan.tokens
		b1 = " ".join(b)
		what_sentence = what_sentence.replace('which', b1)
	return which_sentence

def find_whichType_what(what_sentence, replacedSpan, client):
	c = client.annotate(what_sentence)
	what_parse_tree = c.sentence[0].parseTree
	Q = queue.Queue(maxsize=MAX_SIZE)
	Q.put(what_parse_tree)
	tokens_list = parse_tree_WHNP(Q)
	spanList = generateSpans(tokens_list, c.sentence[0].token)
	index = 0
	for j in range(len(spanList)):
		ts = " ".join(spanList[j].tokens)
		if 'what' in ts:
			index = j
			break;

	if spanList:
		a = spanList[index].tokens
		a1 = " ".join(a)
		b = replacedSpan.tokens
		b1 = " ".join(b)
		try:
			what_sentence = what_sentence.replace(a1, b1)
		except Exception as e:
			pass
	else:
		b = replacedSpan.tokens
		b1 = " ".join(b)
		what_sentence = what_sentence.replace('what', b1)
	return what_sentence

'''
#['Which', 'country', "'s", 'courts']O
#['the', 'ECHR']ORGANIZATION
#['a', 'wider', 'stance']O
#['provisions']O
#['genocide', 'laws']O
def testPart():
	TEST_SENTENCE = "what religious activity be responsible for the grow demand for hostelry"
	with corenlp.CoreNLPClient(annotators='tokenize ssplit parse lemma pos ner'.split()) as testclient:
		c = testclient.annotate(TEST_SENTENCE)
		TEST_PARSE_TREE = c.sentence[0].parseTree
	TEST_Q = queue.Queue(maxsize=MAX_SIZE)
	TEST_Q.put(TEST_PARSE_TREE)
	print (TEST_PARSE_TREE)
	tokens_list = parse_tree_WHNP(TEST_Q)
	print (tokens_list)

	spanList = generateSpans(tokens_list, c.sentence[0].token)
	sizeList = len(spanList)
	for s in spanList:
		print (str(s.tokens) + str(s.type))
	return 0
testPart()
'''
def getBestAnswer(spanList, parsedQuestion, context, model, questionType, client):
	score = []
	indexArray = []
	resultIndex = 0

	#WHO : PERSON NER
	if questionType == 1:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			if spanType == 'PERSON':
				strReplaced = parsedQuestion.replace('who', " ".join(tokens))
				x = model.fit_transform([context, strReplaced])
				matrix = (x * x.T).A
				score.append(matrix[0][1])
				indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)

	elif questionType == 2:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			if spanType == 'LOCATION' or spanType == 'ORGANIZATION':
				strReplaced = parsedQuestion.replace('where', " ".join(tokens))
				x = model.fit_transform([context, strReplaced])
				matrix = (x * x.T).A
				score.append(matrix[0][1])
				indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)

	elif questionType == 3:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			if spanType in TIMENER:
				#replace
				strReplaced = parsedQuestion.replace('when', " ".join(tokens))
				x = model.fit_transform([context, strReplaced])
				matrix = (x * x.T).A
				score.append(matrix[0][1])
				indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)

	elif questionType == 6:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			strReplaced = find_which(parsedQuestion, s, client)
			x = model.fit_transform([context, strReplaced])
			matrix = (x * x.T).A
			score.append(matrix[0][1])
			indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)

	elif questionType == 7:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			strReplaced = find_whichType_what(parsedQuestion, s, client)
			x = model.fit_transform([context, strReplaced])
			matrix = (x * x.T).A
			score.append(matrix[0][1])
			indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)

	elif questionType == 4:
		#how many OR how much
		if "how many" in parsedQuestion:
			for j in range(len(spanList)):
				s = spanList[j]
				tokens = s.tokens
				spanType = s.type
				if spanType in NUMERICAL:
					#replace
					strReplaced = parsedQuestion.replace('how many', " ".join(tokens))
					x = model.fit_transform([context, strReplaced])
					matrix = (x * x.T).A
					score.append(matrix[0][1])
					indexArray.append(j)
		elif "how much" in parsedQuestion:
			for j in range(len(spanList)):
				s = spanList[j]
				tokens = s.tokens
				spanType = s.type
				if spanType in NUMERICAL:
					#replace
					strReplaced = parsedQuestion.replace('how many', " ".join(tokens))
					x = model.fit_transform([context, strReplaced])
					matrix = (x * x.T).A
					score.append(matrix[0][1])
					indexArray.append(j)
		elif "how long" in parsedQuestion:
			for j in range(len(spanList)):
				s = spanList[j]
				tokens = s.tokens
				spanType = s.type
				if spanType in NUMERICAL or spanType in TIMENER:
					#replace
					strReplaced = parsedQuestion.replace('how long', " ".join(tokens))
					x = model.fit_transform([context, strReplaced])
					matrix = (x * x.T).A
					score.append(matrix[0][1])
					indexArray.append(j)
		elif "how old" in parsedQuestion:
			for j in range(len(spanList)):
				s = spanList[j]
				tokens = s.tokens
				spanType = s.type
				if spanType in NUMERICAL or spanType in TIMENER:
					#replace
					strReplaced = parsedQuestion.replace('how old', " ".join(tokens))
					x = model.fit_transform([context, strReplaced])
					matrix = (x * x.T).A
					score.append(matrix[0][1])
					indexArray.append(j)
		elif "how far" in parsedQuestion:
			for j in range(len(spanList)):
				s = spanList[j]
				tokens = s.tokens
				spanType = s.type
				if spanType in NUMERICAL:
					#replace
					strReplaced = parsedQuestion.replace('how far', " ".join(tokens))
					x = model.fit_transform([context, strReplaced])
					matrix = (x * x.T).A
					score.append(matrix[0][1])
					indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)
	elif questionType == 0:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			strReplaced = parsedQuestion.replace('what', " ".join(tokens))
			x = model.fit_transform([context, strReplaced])
			matrix = (x * x.T).A
			score.append(matrix[0][1])
			indexArray.append(j)
		scoreArray = np.array(score)
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)
	else:
		resultIndex = random.randint(0, len(spanList) - 1)
	return resultIndex


#use corenlp to tokenize ssplit lemma pos ner
with corenlp.CoreNLPClient(annotators='tokenize ssplit parse lemma pos ner'.split()) as client:
	for p in line['data']:
		new_para = defaultdict(list)
		paras = p['paragraphs']
		for section in paras:
			raw_context = section['context']
			set_qass = section['qas']

			#parse context
			#get lemmas of words, remove punctuations
			temp_context = client.annotate(raw_context)
			parsed_context = []
			for s in temp_context.sentence:
				this_sentence = []
				for token in s.token: 
					ts = token.lemma.lower()
					if ts not in PUNCTUATIONS:
						this_sentence.append(ts)
				parsed_sentence = " ".join(this_sentence)
				parsed_context.append(parsed_sentence)

			#train TFIDF model using context content 
			unigram_model = TFIDF(input=parsed_context, analyzer='word', dtype=np.float32, stop_words=STOP_WORDS)			
			#deal with questions
			for q in set_qass:
				raw_question = q['question']
				qid = q['id']
				
				#parse question
				#get lemmas of words remove punctuations
				#get the question type
				#state 1: temp_question.sentence[0].token
				#state 2: np.array(this_question)
				temp_question = client.annotate(raw_question)
				parsed_question = []
				this_question = []
				size_tokens = len(temp_question.sentence[0].token)
				list_tokens = temp_question.sentence[0].token
				IDENTFY = False
				qtype = 8
				for index in range(size_tokens):
					qs = list_tokens[index].lemma.lower()
					if qs not in PUNCTUATIONS:
						this_question.append(qs)
						if not IDENTFY:
							if qs == 'what':
								IDENTFY = True
								qtype_pos = list_tokens[index].pos
								if qtype_pos == 'WP':
									#noun what
									qtype = 0
								elif qtype_pos == 'WDT':
									#which what
									qtype = 7
								else:
									IDENTFY = False
							elif qs == 'who':
								IDENTFY = True
								qtype = 1
							elif qs == 'where':
								IDENTFY = True
								qtype = 2
							elif qs == 'when':
								IDENTFY = True
								qtype = 3
							elif qs == 'how':
								IDENTFY = True
								qtype = 4
							elif qs == 'why':
								IDENTFY = True
								qtype = 5
							elif qs == 'which':
								IDENTFY = True
								qtype = 6
				parsed_question_str = " ".join(this_question)
				parsed_question.append(parsed_question_str)

				#get the TFIDF of the question
				find_max_smilarity = []
				for j in range(len(parsed_context)):
					a_c = parsed_context[j]
					a_q = parsed_question_str
					combine = [a_c, a_q]
					matrix = unigram_model.fit_transform(combine)
					scoreA = (matrix * matrix.T).A
					find_max_smilarity.append(scoreA[0][1])
				find_max_smilarity = np.array(find_max_smilarity)
				max_index_sentence = np.argmax(find_max_smilarity)

				#HAVE found the sentence containing the answer
				#index : max_index_sentence
				candidate_sentence = temp_context.sentence[max_index_sentence]
				tree = candidate_sentence.parseTree
				Q = queue.Queue(maxsize=MAX_SIZE)
				Q.put(tree)
				canditate_answers = parse_tree(Q)

				#generate candidate spans
				#e.g.['The', 'writings']O,['the', 'pub']O,['Samuel', 'Pepys']PERSON,['the', 'heart']O,['England']LOCATION
				spanList = generateSpans(canditate_answers, candidate_sentence.token)
				sizeList = len(spanList)

				#parsed_sentence
				#y_context[max_index_sentence]
				#unigram_model
				#type
				#size_vocabulary
				resultIndex = getBestAnswer(spanList, parsed_question_str, parsed_context[max_index_sentence], unigram_model, qtype, client)
				#who
				#find PERSON NER
				answerString = ""
				s = spanList[resultIndex]
				for j in range(len(s.tokens) - 1):
					answerString += s.tokens[j] + " "
				answerString += s.tokens[len(s.tokens) - 1]
				ANSWER_SET[qid] = answerString


result_json = json.dumps(ANSWER_SET)
with open(OUTPUTFILE, 'w') as fout:
	fout.writelines(result_json)			





