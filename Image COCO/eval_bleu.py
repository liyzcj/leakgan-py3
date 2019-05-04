import nltk
import random
from scipy import stats
# import cPickle
import pickle


data_Name = "cotra"
vocab_file = "tmp/vocab_" + data_Name + ".pkl"
vocab_file_py3 = "tmp/vocab_" + data_Name + "_py3.pkl"
with open(vocab_file_py3, 'rb') as f:
    word, vocab = pickle.load(f)
# with open(vocab_file, 'r') as f:
#     word, vocab = cPickle.load(f)

# with open(vocab_file_py3, 'wb') as f:
#     cPickle.dump((word, vocab), f)

pad = vocab[' ']
print (pad)

samples_path = './ckpts/test2/samples'

reference_file = 'tmp/realtest_coco.txt'
# hypothesis_file_leakgan = 'save/generator_sample.txt'
hypothesis_file_leakgan = samples_path + '/coco_81.txt'
#################################################
reference = []
with open(reference_file)as fin:
    for line in fin:
        candidate = []
        line = line.split()
        for i in line:
            if i == str(pad):
                break
            candidate.append(i)

        reference.append(candidate)
#################################################
hypothesis_list_leakgan = []
with open(hypothesis_file_leakgan) as fin:
    for line in fin:
        line = line.split()
        while line[-1] == str(pad):
            line.remove(str(pad))
        hypothesis_list_leakgan.append(line)
#################################################
#################################################
random.shuffle(hypothesis_list_leakgan)
#################################################

for ngram in range(2,6):
    weight = tuple((1. / ngram for _ in range(ngram)))
    bleu_leakgan = []
    bleu_supervise = []
    bleu_base2 = []
    num = 0
    for h in hypothesis_list_leakgan[:2000]:
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, h, weight)
        # print (num, BLEUscore)
        num += 1
        bleu_leakgan.append(BLEUscore)
    print ('leakgan')
    print(len(weight), '-gram BLEU score : ', 1.0 * sum(bleu_leakgan) / len(bleu_leakgan))

# cPickle.dump([hypothesis_list_leakgan], open('save/significance_test_sample.pkl', 'w'))
