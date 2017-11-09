import corenlp
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import numpy as np
import json
import queue
import random

#external libraries: 
#StanfordCoreNLP, NLTK, sklearn.feature_extraction.text
#other libraries: 
#defaultdict, numpy, json, queue, random

nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = nltk.corpus.stopwords.words('english')
PUNCTUATIONS = [',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%']
QUESTIONTYPE = ['what','who','where','when','how','why','which','other']
TIMENER = ['DATE','DURATION','TIME']
NUMERICAL = ['MONEY','NUMBER','ORDINAL','PERCENT']
MAX_SIZE = 100

#ANSWER_SET is a dictionary which stores [questionID,Corresponding Answer]
#e.g. ["5733be284776f4190066117f": "Venite Ad Me Omnes"]
ANSWER_SET = defaultdict(str)
OUTPUTFILE = 'result.json'
INPUTFILE = 'singleparagraph.json'

#READ JSON File
file = INPUTFILE
with open(file, 'r') as f:
	data = f.read()
line = json.loads(data)

#a class for each question, recording its type, question content and part for substitution
class questionSpan:
	def __init__(self):
		self.qtype = 8
		self.replacedPart = ""
		self.question = ""

#a class for each candidate answer, recording its tokens and type
class candidateAnswerSpan:
	def __init__(self):
		self.tokens = []
		self.type = 'O'

class Span:
	def __init__(self):
		self.tokens = []

#Get tokens corresponding to an NP non-terminals from a document parse Tree 
#e.g. given [NP([NP([DT([The])],[NNS([writings])])],[PP([IN([of])],[NP([NNP(Samuel)],[NNP(Pepys)])])])]
#e.g. given [NP([NNP(Samuel)],[NNP(Pepys)])]
#the function will return [Samuel Pepys] as an integration
def getNP_fromTree(tree):
	result = ""
	if len(tree.child) > 0:
		for child in tree.child:
			if child.value == "NP":
				return "", True
			t, h = getNP_fromTree(child)
			if h:
				return "", True
			else:
				result += t + " "
	else:
		return tree.value, False
	return result, False

#Get tokens corresponding to an WHNP non-terminals from a document parse Tree 
#e.g. given child [WHNP([WDT(Which)],[NN(prize)])]
#the fucntion will return [Which prize] as an integration
def getWHNP_fromTree(tree):
	result = ""
	if len(tree.child) > 0:
		for child in tree.child:
			if child.value == "WHNP":
				return "", True
			t, h = getWHNP_fromTree(child)
			if h:
				return "", True
			else:
				result += t + " "
	else:
		return tree.value, False
	return result, False

#Process the parse tree to get all NP tokens
#e.g. given a sentence that "The writings of Samuel Pepys describe the pub as the heart of England."
#the function will return [[The writings], [the pub], [Samuel Pepys], [the heart], [England]] consisting of 5 NP 
def parse_tree_NP(Q):
	#BFS: Find all NP subtrees
	tokens_q = queue.Queue(maxsize=MAX_SIZE)
	while not Q.empty():
		tree = Q.get()
		if tree.value == 'NP':
			tokens_q.put(tree)
		if len(tree.child) > 0:
			for child in tree.child:
				Q.put(child)
	
	#Get all conrresponding tokens 
	tokens_list = []
	while not tokens_q.empty():
		tr = tokens_q.get()
		(r, h) = getNP_fromTree(tr)
		if not h:
			tl = r.split()
			tokens_list.append(tl)
	return tokens_list

#Process the parse tree to get all WHNP tokens
#e.g. given a sentence that "Which prize did Frederick Buechner create?"
#the function will return [Which prize] which is a single WHNP
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

#Generate a candidate answer consists of answer type and answer tokens
#e.g. ["Samuel", "Pepys"] PERSON
#e.g. ["the", "pub"] O
def generate_candidateAnswerSpans(l1, l2):
	count = defaultdict(int)
	span_list = []
	for s in l1:
		sp = candidateAnswerSpan()
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

def generate_Spans(l1, l2):
	count = defaultdict(int)
	span_list = []
	for s in l1:
		sp = Span()
		for word in s:
			sp.tokens.append(word)
			for wordl2 in l2:
				w = wordl2.word
				if word == w:
					break;
		span_list.append(sp)
	return span_list

#Since when we attempt to select a best answer from candidates 
#we will replace tokens in WH- or How with each candidates to get similarity score using tf-idf,
#we need to find a valid part to replace.
#e.g. Question: "Which prize did Frederick Buechner create?"
#The function will return "Which prize".
#Then we will replace "which prize " with each candidates, then we get a new sentence "Buechner Prize for Preaching did Frederick Buechner create"
def generateReplacedPart(parsedQuestion, qtype):
	rp = ""
	Q = queue.Queue(maxsize=MAX_SIZE)
	parseT = parsedQuestion.parseTree
	Q.put(parseT)
	tokens_list = parse_tree_WHNP(Q)
	spanList = generate_Spans(tokens_list, parsedQuestion.token)
	#WHICH
	if qtype == 6:
		rp = "which"
		index = 0
		for j in range(len(spanList)):
			ts = " ".join(spanList[j].tokens)
			if 'which' in ts:
				index = j
				break;
		if spanList:
			a = spanList[index].tokens
			rp = " ".join(a)
		else:
			rp = "which"
	#WDT WHAT
	elif qtype == 7:
		rp = "what"
		index = 0
		for j in range(len(spanList)):
			ts = " ".join(spanList[j].tokens)
			if 'what' in ts:
				index = j
				break;
		if spanList:
			a = spanList[index].tokens
			rp = " ".join(a)
		else:
			rp = "what"	
	return rp


# just a test part
def testPart():
	TEST_SENTENCE = "Which prize did Frederick Buechner create?"
	with corenlp.CoreNLPClient(annotators='tokenize ssplit parse lemma pos ner'.split()) as testclient:
		c = testclient.annotate(TEST_SENTENCE)
		TEST_PARSE_TREE = c.sentence[0].parseTree
	TEST_Q = queue.Queue(maxsize=MAX_SIZE)
	TEST_Q.put(TEST_PARSE_TREE)
	print (TEST_PARSE_TREE)
	tokens_list = parse_tree_WHNP(TEST_Q)
	print (tokens_list)

	spanList = generate_candidateAnswerSpans(tokens_list, c.sentence[0].token)
	sizeList = len(spanList)
	for s in spanList:
		print (str(s.tokens) + str(s.type))
	return 0
#testPart()

#Select the best answer from candidate answer list
#According to different question types, we will generate corresponding sublists of the original candidate answer lists
#e.g. for WHO question, we will add all spans whose NER is PERSON to the sublist.
#Then for each span in the sublist, substitute the conrresponding interrogative pronoun with them, thus generating a new sentence.
#Finally we calculate TF-IDF scores separately for each new sentence, and we will select the final answer which is of the hightest score.
def getBestAnswer(spanList, QSpan, context, model):
	score = []
	indexArray = []
	resultIndex = 0

	questionType = QSpan.qtype
	parsedQuestion = QSpan.question

	#WHO : PERSON NER
	if questionType == 1:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			if spanType == 'PERSON':
				strReplaced = parsedQuestion.replace('who', " ".join(tokens))
				#calculate tf-idf
				x = model.fit_transform([context, strReplaced])
				matrix = (x * x.T).A
				score.append(matrix[0][1])
				indexArray.append(j)
		scoreArray = np.array(score)
		#select one of the highest score
		if len(scoreArray) > 0:
			i = np.argmax(scoreArray)
			resultIndex = indexArray[i]
		else:
			resultIndex = random.randint(0, len(spanList) - 1)

	#WHERE: LOCATION OR ORGANIZATION NER
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

	#WHEN: TIMENER
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

	#WHICH
	elif questionType == 6:
		for j in range(len(spanList)):
			s = spanList[j]
			tokens = s.tokens
			spanType = s.type
			strReplaced = parsedQuestion.replace(QSpan.replacedPart, " ".join(tokens))
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
			strReplaced = parsedQuestion.replace(QSpan.replacedPart, " ".join(tokens))
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

	#HOW
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

	#WHAT
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


#Use stanford corenlp to tokenize given documents including contexts and questions, then split tokens into sentences.
#Obtain corresponding parse tree, lemma, POS, NER.
with corenlp.CoreNLPClient(annotators='tokenize ssplit parse lemma pos ner'.split()) as client:
	for p in line['data']:
		new_para = defaultdict(list)
		paras = p['paragraphs']
		for section in paras:
			raw_context = section['context']
			set_qass = section['qas']

			#process context
			#filter out punctuations 
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

			#train TFIDF model using processed context content without stopwords
			unigram_model = TFIDF(input=parsed_context, analyzer='word', dtype=np.float32, stop_words=STOP_WORDS)			
			#process each question in the question & answers set
			for q in set_qass:
				raw_question = q['question']
				qid = q['id']
				
				#process a single question, generating a corresponding questionSpan
				#additionally, analize the type of the question
				#if the question type is WDT WHAT or WHICH, we should get the part for substitution
				temp_question = client.annotate(raw_question)
				this_question = []
				size_tokens = len(temp_question.sentence[0].token)
				list_tokens = temp_question.sentence[0].token
				IDENTFY = False
				qtype = 8
				QSpan = questionSpan()
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
									QSpan.replacedPart = generateReplacedPart(temp_question.sentence[0], qtype)
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
								QSpan.replacedPart = generateReplacedPart(temp_question.sentence[0], qtype)
				parsed_question = " ".join(this_question)
				QSpan.qtype = qtype				
				QSpan.question = parsed_question


				#calculate the TFIDF of the question using unigram model trained as above
				#we use this to evaluate the similarity between given question and each sentence of context
				#max_index_sentence records the index of sentence which has best smilarity performance
				find_max_smilarity = []
				for j in range(len(parsed_context)):
					a_c = parsed_context[j]
					a_q = parsed_question
					combine = [a_c, a_q]
					matrix = unigram_model.fit_transform(combine)
					scoreA = (matrix * matrix.T).A
					find_max_smilarity.append(scoreA[0][1])
				find_max_smilarity = np.array(find_max_smilarity)
				max_index_sentence = np.argmax(find_max_smilarity)

				#HAVE found the sentence containing candidate answers
				#index : max_index_sentence
				#use function parse_tree_NP() to find all NPs in this sentence
				candidate_sentence = temp_context.sentence[max_index_sentence]
				tree = candidate_sentence.parseTree
				Q = queue.Queue(maxsize=MAX_SIZE)
				Q.put(tree)
				canditate_answers = parse_tree_NP(Q)

				#generate candidate answer spans with their corresponding types
				#e.g.['The', 'writings']O,['the', 'pub']O,['Samuel', 'Pepys']PERSON,['the', 'heart']O,['England']LOCATION
				spanList = generate_candidateAnswerSpans(canditate_answers, candidate_sentence.token)
				sizeList = len(spanList)

				#use function getBestAnswer() to find the best answer from candidate answer list
				resultIndex = getBestAnswer(spanList, QSpan, parsed_context[max_index_sentence], unigram_model)
				answerString = ""
				s = spanList[resultIndex]

				#generate the final answer string based on the best answer span generated above
				for j in range(len(s.tokens) - 1):
					answerString += s.tokens[j] + " "
				answerString += s.tokens[len(s.tokens) - 1]
				ANSWER_SET[qid] = answerString

#generate a JSON file consisting of results
result_json = json.dumps(ANSWER_SET)
with open(OUTPUTFILE, 'w') as fout:
	fout.writelines(result_json)			

