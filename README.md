# QA System
It's a project for Natural Language Processing (CS 5740)

Goal: We are required to apply NLP knowledge to implement a QA system that will work on a real world challenge task: SQuAD. No human intervention is allowed in deriving answers.
Input: a passage with a set of questions.
Output: for each question, output a segment of text from the passage that answers it.

Implement: we tried two different approaches for the QA system
  * Project3_Shuo.ipynb [logistic regression] 
  * xyproject3.py [unigram model]

HOW TO RUN xyproject3.py
corenlp.CoreNLPClient(annotators='tokenize ssplit parse lemma pos ner'.split()), TfidfVectorizer(), generateReplacedPart(), unigram_model.fit_transform(), parse_tree_NP(), generate_candidateAnswerSpans(), getBestAnswer()

functions:
1.getNP_fromTree(tree): Get tokens corresponding to an NP non-terminals from a document parse Tree.

2.getWHNP_fromTree(tree): Get tokens corresponding to an WHNP non-terminals from a document parse Tree

3.parse_tree_NP(Q): Process the parse tree to get all NP tokens.

4.parse_tree_WHNP(Q): Process the parse tree to get all WHNP tokens.

5.generate_candidateAnswerSpans(l1, l2): Generate a candidate answer consists of answer type and answer tokens

6.generate_Spans(l1, l2): 

7.generateReplacedPart(parsedQuestion, qtype): Since when we attempt to select a best answer from candidates we will replace tokens in WH- or How with each candidates to get similarity score using tf-idf, we need to find a valid part to replace.

8.getBestAnswer(spanList, QSpan, context, model): Select the best answer from candidate answer list

9.NOT A FUNCTION just the __init__ code: 
(1)Use stanford corenlp to tokenize given documents including contexts and questions, then split tokens into sentences. Obtain corresponding parse tree, lemma, POS, NER.
(2)Process context content, filtering out pounctuations and stopwords.
(3)Train a unigram TF-IDF model using parsed context as input.
(4)Process a single question, generating a corresponding question span. Additionally, give the type of question.
(5)Calculate the TFIDF of the question using unigram model trained as above.
(6)Use function parse_tree_NP() to find all NPs in this sentence.
(7)Generate candidate answer spans with their corresponding types.
(8)Output answers.
