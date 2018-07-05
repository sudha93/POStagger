import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
# we are taking penn treebank corpus from nltk , below line extracts tagged 
# sentences 
# training our own models using gensim
# the input should given in the form of list of lists 
tagged_sentences = nltk.corpus.treebank.tagged_sents()
print len(tagged_sentences)
sentences = []
for sent in tagged_sentences:
	tsent=[]
	for index in range(len(sent)):
		if index==0 :
			tsent.append("<start>")
		
		tsent.append(sent[index][0])
		if index== len(sent)-1:
			tsent.append("<end>")
	sentences.append(tsent)

print "training started"
model = Word2Vec(sentences, min_count=1)
print "model loaded"

def getfeatures(sent, ind): #every wordvec is in the form of numpy array 
	d={
		#'word':sent[ind][0],
		'wordemb':model.wv[sent[ind][0]],
		'first':np.array([1.0]) if ind==0 else np.array([-1.0]),
		'last':np.array([1.0]) if ind==len(sent)-1 else np.array([-1.0]),
		'pword': model.wv['<start>'] if ind==0 else model.wv[sent[ind-1][0]],
		'nword': model.wv['<end>'] if ind==len(sent)-1 else model.wv[sent[ind+1][0]]
		}
	#print d['first']
	#tp =  np.concatenate((d['wordemb'],d['first']), axis=0 )
	#tp = np.concatenate((tp, d['last']), axis=0)
	tp = d['wordemb']	
	tp = np.concatenate((tp, d['pword']), axis =0)
	tp = np.concatenate((tp, d['nword']), axis=0)
	return tp  # tp is a concatenation of vectors

def gettag(sent, ind):
	return sent[ind][1]


features = []
tags = []
for sentence in tagged_sentences:
	for i in range(len(sentence)):
		features +=  [getfeatures(sentence, i)]  # list of vectors
		tags += [gettag(sentence, i)]    # getting the target tag and pushing them into a list 
#exit()
#print features[:2]
#print tags[:10]

#v = DictVectorizer()
#inp = v.fit_transform(features)

lb = preprocessing.LabelBinarizer()
lb.fit(tags)    # just creates indices
out = lb.transform(tags)	# converts tags to one hot vectors according to above indexing
#print out[:4]

cutoff = int(0.7*len(features))	

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, verbose=True) 
print 'training'
clf.fit(features[:cutoff], out[:cutoff])
print 'results'
print clf.score(features[cutoff:], out[cutoff:])





