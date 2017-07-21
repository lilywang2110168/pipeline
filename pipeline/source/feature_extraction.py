import codecs
import os

import nltk
from nltk.stem import WordNetLemmatizer

# global variabes
ps = nltk.stem.PorterStemmer()
lemmatizer = WordNetLemmatizer()
# stopwords for unigrams
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
# stop words for bigrams
stopPhrases = set(
    ['price range', 'price point', 'star rating', 'quality product', 'great product', 'good product', 'price tag'])
brandSet = set()
# the brand file
path = os.path.join(os.path.dirname(__file__), '../resources/laptop_brands.txt')
with codecs.open(path, "r", encoding="utf-8", errors='ignore') as f2:
    for line in f2:
        brand = line.strip()
        brandSet.add(brand)


def getUnigrams(tokens):
    dictionary = {}
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

    return dictionary


def getBigrams(result):
    dictionaryPhrases = {}

    for sent in result:
        for word in sent:
            if type(word) is nltk.Tree:
                tmp = word.leaves()

                # lemmatizing
                word1 = lemmatizer.lemmatize(tmp[0][0])
                word2 = lemmatizer.lemmatize(tmp[1][0])
                word1 = word1.encode('ascii', 'ignore').lower()
                word2 = word2.encode('ascii', 'ignore').lower()

                oriPhrase = word1 + ' ' + word2
                ori2 = word1 + word2

                if oriPhrase not in stopPhrases and oriPhrase not in brandSet and ori2 not in brandSet and word1 not in stopwords:

                    # stemming
                    word1 = ps.stem(word1)
                    word2 = ps.stem(word2)
                    word1 = word1.encode('ascii', 'ignore').lower()
                    word2 = word2.encode('ascii', 'ignore').lower()

                    tmp = word1 + ' ' + word2

                    tmp2 = word1 + word2

                    if tmp not in stopPhrases and tmp not in brandSet and tmp2 not in brandSet:
                        if tmp in dictionaryPhrases:
                            dictionaryPhrases[tmp]['num'] = dictionaryPhrases[tmp]['num'] + 1
                            if oriPhrase in dictionaryPhrases[tmp]:
                                dictionaryPhrases[tmp][oriPhrase] = dictionaryPhrases[tmp][oriPhrase] + 1
                            else:
                                dictionaryPhrases[tmp][oriPhrase] = 1


                        else:
                            dictionaryPhrases[tmp] = {'num': 1, oriPhrase: 1}

    return dictionaryPhrases


def pruneFeature(dictionary, dictionaryPhrases):
    deleteSingle = []
    deletePhrase = []

    for key in dictionaryPhrases:

        # dealing with 'laptop' and 'lap top'
        numPhrase = dictionaryPhrases[key]['num']

        [word1, word2] = key.split()
        keyUnigram = word1 + word2

        if keyUnigram in dictionary:
            numUnigram = dictionary[keyUnigram]['num']

            if (numPhrase >= numUnigram):
                deleteSingle.append(keyUnigram)
                dictionaryPhrases[key]['num'] = numPhrase + numUnigram

            else:

                deletePhrase.append(key)
                dictionary[keyUnigram]['num'] = numUnigram + numPhrase
                if word1 in dictionary:
                    deleteSingle.append(word1)
                if word2 in dictionary:
                    deleteSingle.append(word2)


                    # dealing with 'battery', 'battery life' and 'life'
        if word1 in dictionary and word2 in dictionary:

            num1 = dictionary[word1]['num']
            num2 = dictionary[word2]['num']
            numPhrase = dictionaryPhrases[key]['num']

            if numPhrase > num1 / 8:
                dictionaryPhrases[key]['num'] = numPhrase * 2
                deleteSingle.append(word1)

            if numPhrase > num2 / 8:
                dictionaryPhrases[key]['num'] = numPhrase * 2
                deleteSingle.append(word2)

    return deleteSingle, deletePhrase


# takes in a dictionary of stemmed words and returns a dictionary of the most representative features
def getRepresentativeFeatures(dict, num):
    dictionary = {}

    for key, value in dict.iteritems():
        if value['num'] > num:
            max = 0
            maxWord = ''

            for word in value:
                if word != 'num':
                    if value[word] > max:
                        max = value[word]
                        maxWord = word

            dictionary[maxWord] = max
    return dictionary


# this function gets the top features from the dictionaries
# returns a list of tuples (feature, freaquency)
def getTopFeatures(dict, num):
    myList = []
    i = 0
    for key, value in reversed(sorted(dict.iteritems(), key=lambda (k, v): (v, k))):
        if i > num:
            break
        else:
            i = i + 1
            myList.append((key, value))

    return myList
