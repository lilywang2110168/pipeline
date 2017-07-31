import nltk
import pyspark
from multiprocessing import Pool
import time

from feature_extraction import ( getUnigrams, getBigrams, pruneFeature, getRepresentativeFeatures,
                                                getTopFeatures)
from spark import (get_sc, load_table)


start_time = time.time()
sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

#globals
ps = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()


load_table(spark, 'AmazonReviews')
df = spark.sql('SELECT reviewText from AmazonReviews')

stopwords = set(
    ["a", "item", "great", "good", "excellent", "nice", "long", "first", "new", "bit", "side", "everything", "review",
     "piece", "feel", "pair", "a's", "color", "able", "about", "above", "issue", "product", "feature", "week", "money",
     "problem", "year", "according", "accordingly", "work", "day", "across", "actually", "after", "afterwards", "again",
     "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always",
     "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "hour",
     "month", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't",
     "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became",
     "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below",
     "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
     "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co",
     "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
     "contains", "corresponding", "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite",
     "did", "didn't", "different", "do", "does", "doesn't", "doing", "don't", "done", "down", "downwards", "during",
     "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc",
     "even", "ever", "every", "everybody", "time", "thing", "lot", "everyone", "everything", "everywhere", "ex",
     "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows",
     "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting",
     "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens",
     "hardly", "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here",
     "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither",
     "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate",
     "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into",
     "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept",
     "know", "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let",
     "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may",
     "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my",
     "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
     "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing",
     "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one",
     "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
     "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please",
     "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re",
     "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said",
     "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming",
     "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she",
     "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime",
     "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub",
     "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx",
     "that", "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there's",
     "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd", "they'll",
     "they're", "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through",
     "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly",
     "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
     "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various", "very", "via",
     "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome",
     "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's",
     "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
     "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without",
     "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've",
     "your", "yours", "yourself", "yourselves", "z", "zero"])
dictionary={}
def getUnigrams:
    for sent in tokens:
        for word in sent:
            if word[1] == 'NN' or word[1] == 'NNS':

                # lemmitizing
                tmp = lemmatizer.lemmatize(word[0], pos='n')
                tmp = tmp.encode('ascii', 'ignore').lower()
                oriword = tmp

                # ignoring the stopwords and brand names
                if tmp not in stopwords and tmp not in brandSet:

                    # stemming
                    tmp = ps.stem(tmp)
                    tmp = tmp.encode('ascii', 'ignore').lower()

                    # ignoring the stopwords and brand names
                    if tmp not in stopwords and tmp not in brandSet:

                        if tmp in dictionary:
                            dictionary[tmp]['num'] = dictionary[tmp]['num'] + 1
                            if oriword in dictionary[tmp]:
                                dictionary[tmp][oriword] = dictionary[tmp][oriword] + 1
                            else:
                                dictionary[tmp][oriword] = 1

                        else:
                            dictionary[tmp] = {'num': 1, oriword: 1}





def parseGrammar(sent):
  grammar = r"""
    NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
      """
  cp = nltk.RegexpParser(grammar)
  return cp.parse(sent)
  
grammar = r"""
 NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
"""

cp = nltk.RegexpParser(grammar)


def getSentences(row):
  return str(row.reviewText)
    
pool = Pool(16) 

sentences=pool.map(getSentences, df.collect())
print("2--- %s seconds ---loading data" % (time.time() - start_time))
start_time = time.time()

pool = Pool(16) 
tokens = pool.map(nltk.word_tokenize, sentences)
tokens= pool.map(nltk.pos_tag, tokens)
print("-- %s seconds ---tokenizing and POS tagging" % (time.time() - start_time))
start_time = time.time()
result=pool.map(parseGrammar, tokens)
pool.close() 
pool.join()
print("2--- %s seconds ---grammer" % (time.time() - start_time))
start_time = time.time()
pool.close() 
pool.join() 

getUnigrams()
print("2--- %s seconds ---get unigrams" % (time.time() - start_time))
start_time = time.time()
dictionaryPhrases = getBigrams(result)
print("2--- %s seconds ---get bigrams" % (time.time() - start_time))
start_time = time.time()
deleteSingle, deletePhrase = pruneFeature(dictionary, dictionaryPhrases)

for item in deleteSingle:
    if item in dictionary:
        del dictionary[item]
for item in deletePhrase:
    if item in dictionaryPhrases:
        del dictionaryPhrases[item]
print("2--- %s seconds ---deleting phrases" % (time.time() - start_time))
start_time = time.time()        
        
dictionary = getRepresentativeFeatures(dictionary, 10)
dictionaryPhrases = getRepresentativeFeatures(dictionaryPhrases, 5)

print("2--- %s seconds ---deal with stemming" % (time.time() - start_time))
myList = getTopFeatures(dictionary, 10)
print myList
myList2 = getTopFeatures(dictionaryPhrases, 20)
print myList2
print("2--- %s seconds ---get top features" % (time.time() - start_time))

print("2--- %s seconds ---the rest" % (time.time() - start_time))
                 
'''

##USING SPARK MAP...
sentences = [ str(i.reviewText) for i in df.collect()]
reviews=sc.parallelize(sentences)


#not doing sentence tokenizer
tokens=reviews.map(lambda x:nltk.word_tokenize(x)).map(lambda x:nltk.pos_tag(x)).collect()
result=sc.parallelize(tokens).map(lambda x:cp.parse(x)).collect()


##USING RDD MAP WHICH DOES NOT UTINIZE MULTITHREADING
tokens=df.rdd.map(lambda x:nltk.word_tokenize(str(x.reviewText))).map(lambda x:nltk.pos_tag(x)).collect()
result=tokens.map(lambda x:cp.parse(x))

dictionary = getUnigrams(tokens)
dictionaryPhrases = getBigrams(result)
deleteSingle, deletePhrase = pruneFeature(dictionary, dictionaryPhrases)

print "there"

for item in deleteSingle:
    if item in dictionary:
        del dictionary[item]
for item in deletePhrase:
    if item in dictionaryPhrases:
        del dictionaryPhrases[item]
dictionary = getRepresentativeFeatures(dictionary, 10)
dictionaryPhrases = getRepresentativeFeatures(dictionaryPhrases, 5)
myList = getTopFeatures(dictionary, 10)
print myList
myList2 = getTopFeatures(dictionaryPhrases, 20)
print myList2



reviews = [ str(i.reviewText) for i in df.collect()]
sentences=[]
for line in reviews:
  sents = nltk.sent_tokenize(line)
  for sent in sents:
    sentences.append(sent)
  
reviews=sc.parallelize(sentences)
mydata.map(lambda x: x.split('\t')).\
    map(lambda y: (y[0], y[2], y[1]))

reviews = [ str(i.reviewText) for i in df.collect()]

sentences=[]
for line in reviews:
  sents = nltk.sent_tokenize(line)
  for sent in sents:
    sentences.append(sent)



print "I am here"

tokens = [nltk.word_tokenize(sent) for sent in sentences]
tokens = [nltk.pos_tag(sent) for sent in tokens]

print "now I am here"
grammar = r"""
 NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
"""
cp = nltk.RegexpParser(grammar)
result = [cp.parse(sent) for sent in tokens]
dictionary = getUnigrams(tokens)
dictionaryPhrases = getBigrams(result)
deleteSingle, deletePhrase = pruneFeature(dictionary, dictionaryPhrases)

print "there"

for item in deleteSingle:
    if item in dictionary:
        del dictionary[item]
for item in deletePhrase:
    if item in dictionaryPhrases:
        del dictionaryPhrases[item]
dictionary = getRepresentativeFeatures(dictionary, 10)
dictionaryPhrases = getRepresentativeFeatures(dictionaryPhrases, 5)
myList = getTopFeatures(dictionary, 10)
print myList
myList2 = getTopFeatures(dictionaryPhrases, 20)
print myList2

'''
