import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    ngrams = []

    
    if n is 1:
        ngrams.append(('START',))

   
    RangeEnd = (len(sequence)+1)
    for index_word in range(0, RangeEnd):
        tuple_gram = ()
        for index_gram in range(index_word-n+1, index_word+1):
            word = None
            if index_gram < 0:
                word = 'START'
            elif index_gram >= len(sequence):
                word = 'STOP'
            else:
                word = sequence[index_gram]
            if word:
                tuple_gram = tuple_gram + (word,)
        
        ngrams.append(tuple_gram)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.totalcount = 0

        ##Your code here
        for sequence in corpus: 
            for token in get_ngrams(sequence, 1): 
                if not token in self.unigramcounts: 
                    self.unigramcounts[token] = 0 
                self.unigramcounts[token] += 1 

            for token in get_ngrams(sequence, 2): 
                if not token in self.bigramcounts: 
                    self.bigramcounts[token] = 0 
                self.bigramcounts[token] += 1 

            for token in get_ngrams(sequence, 3): 
                if not token in self.trigramcounts: 
                    self.trigramcounts[token] = 0 
                self.trigramcounts[token] += 1 
                
            self.totalcount += len(sequence)  

        self.bigramcounts[("START", "START")] = self.unigramcounts[("START",)] 

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        tricount = self.trigramcounts.get(trigram, 0.0)
        bicount = float(self.bigramcounts.get(trigram[:-1], 0.0))
        
        return (tricount/bicount) if bicount != 0 else 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        bicount = self.bigramcounts.get(bigram, 0.0)
        unicount = float(self.unigramcounts.get(bigram[:-1], 0.0))

        return (bicount/unicount) if unicount != 0 else 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        unicount = self.unigramcounts.get(unigram, 0.0)
        tricount = float(self.trigramcounts.get(unigram[:-1], 0.0))

        return (unicount/tricount) if tricount != 0 else 0.0
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return           

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        temp_unigram = [] 
        temp_unigram.append(trigram[2]) 
        p_unigram = self.raw_unigram_probability(tuple(temp_unigram)) 

        temp_bigram = [] 
        temp_bigram.append(trigram[1]) 
        temp_bigram.append(trigram[2]) 
        p_bigram = self.raw_bigram_probability(tuple(temp_bigram)) 

        p_trigram = self.raw_trigram_probability(trigram) 

        return lambda1 * p_trigram + lambda2 * p_bigram + lambda3 * p_unigram 
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)

        logprob = 0.0
        for gram in trigrams:
            smoothedprob = self.smoothed_trigram_probability(gram)
            if smoothedprob == 0.0:
                continue
            p = math.log(smoothedprob, 2)
            logprob = logprob + p


        return float(logprob)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum = 0 
        count = 0 
        for sentence in corpus: 
            sum += self.sentence_logprob(sentence) 
            count += len(sentence) 

        sum /= count 

        return float(2, -sum)  


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # print(get_ngrams(["natural", "language", "processing"],1))
    # print(get_ngrams(["natural", "language", "processing"],2))
    # print(get_ngrams(["natural", "language", "processing"],3))
    #
    # print(len(model.unigramcounts))
    # print(len(model.bigramcounts))
    # print(len(model.trigramcounts))
    #
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])
    #
    # print("Unigram: " + str(model.raw_unigram_probability(('department',))))
    # print("Bigram: " + str(model.raw_bigram_probability(('highway', 'department'))))
    # print("Trigram: " + str(model.raw_trigram_probability(('state','highway','department'))))
    #
    # print("Smoothed Trigram: " + str(model.smoothed_trigram_probability(('state','highway','department'))))
    # print("Sentence Log Probability: " + str(model.sentence_logprob('The State Highway department')))
    
    
    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('./hw1_data/ets_toefl_data/train_high.txt', './hw1_data/ets_toefl_data/train_low.txt", "./hw1_data/ets_toefl_data/test_high", "./hw1_data/ets_toefl_data/test_low")
    # print(acc)

