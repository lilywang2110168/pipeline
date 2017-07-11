import nltk
from nltk.stem import WordNetLemmatizer


#global variabes
ps = nltk.stem.PorterStemmer()
lemmatizer = WordNetLemmatizer()

def getUnigrams(tokens):
    dictionary={}
    for sent in tokens:
        for word in sent: 
            if word[1]=='NN' or word[1]=='NNS':               
            
                #lemmitizing
                tmp=lemmatizer.lemmatize(word[0], pos='n')
                tmp=tmp.encode('ascii', 'ignore').lower()
                oriword=tmp
                
                #ignoring the stopwords and brand names
                if tmp not in stopwords and tmp not in brandSet: 

                #stemming
                    tmp=ps.stem(tmp)
                    tmp=tmp.encode('ascii', 'ignore').lower()
               
                    #ignoring the stopwords and brand names
                    if tmp not in stopwords and tmp not in brandSet: 
                
                        if tmp in dictionary: 
                            dictionary[tmp]['num']=dictionary[tmp]['num']+1
                            if oriword in  dictionary[tmp]:
                                dictionary[tmp][oriword]=dictionary[tmp][oriword]+1
                            else:
                                dictionary[tmp][oriword]=1

                        else:
                            dictionary[tmp]={'num':1, oriword:1}

    return dictionary



def getBigrams(result):
    dictionaryPhrases={}

    for sent in result: 
        for word in sent:
            if type(word) is nltk.Tree:
                tmp=word.leaves()
        
                #lemmatizing
                word1=lemmatizer.lemmatize(tmp[0][0])
                word2=lemmatizer.lemmatize(tmp[1][0])
                word1=word1.encode('ascii', 'ignore').lower()
                word2=word2.encode('ascii', 'ignore').lower()

                oriPhrase=word1+' '+word2
                ori2=word1+word2


                if oriPhrase not in stopPhrases and oriPhrase not in brandSet and ori2 not in brandSet and word1 not in stopwords:

                    #stemming
                    word1=ps.stem(word1)
                    word2=ps.stem(word2)
                    word1=word1.encode('ascii', 'ignore').lower()
                    word2=word2.encode('ascii', 'ignore').lower()

                    tmp=word1+' '+word2

                    tmp2=word1+word2

                    if tmp not in stopPhrases and tmp not in brandSet and tmp2 not in brandSet:
                        if tmp in dictionaryPhrases: 
                            dictionaryPhrases[tmp]['num']=dictionaryPhrases[tmp]['num']+1
                            if oriPhrase in  dictionaryPhrases[tmp]:
                                dictionaryPhrases[tmp][oriPhrase]=dictionaryPhrases[tmp][oriPhrase]+1
                            else:
                                dictionaryPhrases[tmp][oriPhrase]=1


                        else:
                            dictionaryPhrases[tmp]={'num':1, oriPhrase:1}

    return dictionaryPhrases





def pruneFeature(dictionary, dictionaryPhrases):
    deleteSingle=[]
    deletePhrase=[]

    for key in dictionaryPhrases: 
   
        #dealing with 'laptop' and 'lap top'   
        numPhrase=dictionaryPhrases[key]['num']

        [word1, word2]=key.split()
        keyUnigram=word1+word2


        if keyUnigram in dictionary: 
            numUnigram=dictionary[keyUnigram]['num']

            if(numPhrase >= numUnigram):
                deleteSingle.append(keyUnigram)
                dictionaryPhrases[key]['num']=numPhrase+numUnigram

            else: 
        
                deletePhrase.append(key)
                dictionary[keyUnigram]['num']= numUnigram + numPhrase
                if word1 in dictionary: 
                    deleteSingle.append(word1) 
                if word2 in dictionary:
                    deleteSingle.append(word2)


    #dealing with 'battery', 'battery life' and 'life'
        if word1 in dictionary and word2 in dictionary: 

            num1=dictionary[word1]['num']
            num2=dictionary[word2]['num']
            numPhrase=dictionaryPhrases[key]['num']


            if numPhrase> num1/8:
                dictionaryPhrases[key]['num']=numPhrase*2
                deleteSingle.append(word1)

            if numPhrase> num2/8:
                dictionaryPhrases[key]['num']=numPhrase*2
                deleteSingle.append(word2)
        
    return deleteSingle, deletePhrase



#takes in a dictionary of stemmed words and returns a dictionary of the most representative features
def getRepresentativeFeatures(dict,num):
    dictionary={}

    for key, value in dict.iteritems():
        if value['num']>num:   
            max=0
            maxWord=''

            for word in value: 
                if word != 'num':
                    if value[word]>max: 
                        max=value[word]
                        maxWord=word

            dictionary[maxWord]=max 
    return dictionary


#this function gets the top features from the dictionaries
#returns a list of tuples (feature, freaquency)
def getTopFeatures(dict, num):
    myList=[]
    i=0
    for key, value in reversed(sorted(dict.iteritems(), key=lambda (k,v): (v,k))):
        if i>num:
            break
        else:
            i=i+1
            myList.append((key,value))
    
    return myList






