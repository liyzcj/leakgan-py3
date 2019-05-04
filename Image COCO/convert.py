import pickle
data_Name = "cotra"
vocab_file = "vocab_" + data_Name + ".pkl"
vocab_file_py3 = "tmp/vocab_" + data_Name + "_py3.pkl"


samples_path = './ckpts/test2/samples'
speech_path = './ckpts/test2/speech'

word, vocab = pickle.load(open(vocab_file_py3, 'rb'))
print (len(word))
input_file = samples_path + '/coco_91.txt'
# input_file = 'save/coco_451.txt'
output_file = speech_path + '/' + data_Name + '_' + input_file.split('_')[-1]
with open(output_file, 'w')as fout:
    with open(input_file)as fin:
        for line in fin:
            #line.decode('utf-8')
            line = line.split()
            #line.pop()
            #line.pop()
            line = [int(x) for x in line]
            line = [word[x] for x in line]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)#.encode('utf-8'))
