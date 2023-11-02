import os
import nltk
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from math import log10,sqrt,pow

corpusroot = './US_Inaugural_Addresses'
doc_df = Counter()
doc_tfdict = {}
doc_weights= {}
normalizers=Counter()
final_list={} 
stemmer = PorterStemmer()

def preproces(token):
    token = token.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(token)
    stopword = stopwords.words('english')
    stemmer = PorterStemmer()
    ftoken = [stemmer.stem(tok) for tok in tokens if tok not in stopword] # after stemming and stopwords removal
    return ftoken

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()
        ftoken = preproces(doc)
        tf = Counter(ftoken)
        doc_tfdict[filename] = tf.copy() #TF values stored in tfdict
        tf.clear()
        doc_df += Counter(list(set(ftoken)))

def getidf(token):
    if doc_df[token] == 0:
        token = str.join(' ', preproces(token))
        return log10(len(doc_tfdict)/doc_df[token]) #getIdf
    else:
        return log10(len(doc_tfdict)/doc_df[token]) #getIdf

def doc_weight(filename, token):
    idf = getidf(token)
    return (1 + log10(doc_tfdict[filename][token])) * idf #Weights

def caluclate_doc_normalizer():
    for filename in doc_tfdict:
        doc_weights[filename] = Counter()
        normalizer = 0
        for token in doc_tfdict[filename]:
            doc_weights[filename][token] = doc_weight(filename, token)
            normalizer += pow(doc_weights[filename][token], 2)
        normalizers[filename] = sqrt(normalizer)

caluclate_doc_normalizer()

def calculate_doc_final():
    for filename in doc_weights:
        for token in doc_weights[filename]:
            doc_weights[filename][token] = doc_weights[filename][token] / normalizers[filename] #normalized weights
            if token not in final_list:
                final_list[token] = Counter()
            final_list[token][filename] = doc_weights[filename][token] # each token

calculate_doc_final()

def getweight(filename,token):
    token = str.join(' ', preproces(token))
    return doc_weights[filename][token]

def cosinesimilarity(document, query_normalizer, query_tf):
    answer=[]
    weight = []
    cosinesimilarities=Counter()
    for doc in doc_weights:
        cosinesimilarity = 0
        for token in query_tf:
            if doc in document[token]:
                cosinesimilarity += (query_tf[token] / query_normalizer) * final_list[token][doc]
        cosinesimilarities[doc] = cosinesimilarity
    max =  cosinesimilarities.most_common(1)  # extract maximum cosine similarity
    answer, weight = zip(*max)
    return answer, weight

def query(qstring):
    query = stemmer.stem(qstring)
    qtf = {}
    qnormalizer = 0
    document = {}
    for token in query:
        if token not in final_list:          
            return (None, 0.000)            #return (None,0.00) if not present in any doc
        if getidf(token)==0:
            document[token], weight = zip(*final_list[token].most_common(1))   #Max weight
        else:
            document[token],weight = zip(*final_list[token].most_common(10))   #top 10

        if(qstring.count(token)) == 0:
            qtf[token]=0
        else:
            qtf[token]=1+log10(qstring.count(token))
        qnormalizer+=pow(qtf[token],2)
    qnormalizer=sqrt(qnormalizer)
    out_doc, weight = cosinesimilarity(document, qnormalizer, qtf) #Calculate Similarity

    return out_doc[0],weight[0]

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))