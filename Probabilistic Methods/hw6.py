from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepareDictionary(file):
    g = open(file)
    count = 1
    dict = {}
    for line in g:
        newLine = line.strip()
        dict[count] = newLine
        count += 1
    g.close()
    return(dict)


def discoverTopics(dictionary):
    mm = corpora.MmCorpus('corpus/docword.kos.txt')
    #run with default number of topics (100)
    print('Running with default number of topics')
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary)
    lda.print_topics(5)
    for i in range(5, 21, 15):
        print('Running with', i, 'topics')
        lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=i, passes=5)
        lda.print_topics(5)


def runIt():
    dictionary = prepareDictionary('corpus/vocab.kos.txt')
    discoverTopics(dictionary)

runIt()