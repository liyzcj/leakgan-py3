import numpy as np



# 将句子分段, 并从判别器获取奖励
def get_rewords_from_discriminator(sess, input_x, discriminator, seq_length, batch_size, padding_value):
  """computing rewards for a batch of sentenses
  Input:
      sess: a TensorFlow Session
      input_x: [batch_size x seq_length], a batch of generated sentences
      discriminator: a discriminator object
  Return:
      rewards: the rewards of input_x, [batch_size x seq_length]
  """
  rewards = []  # batch_size x seq_length
  split_data = split_sentence(input_x, seq_length, padding_value) # split data as [SEQ_LENGTH*BATCH_SIZE, SEQ_LENGTH]
  for given_num in range(1, seq_length+1):
    batch_data = [] # batch_size x seq_length
    for i in range(batch_size):
      batch_data.append(split_data[i*seq_length+given_num-1])
    feed = {discriminator.D_input_x: batch_data, discriminator.dropout_keep_prob: 1.0}
    ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
    ypred = np.array([item[1] for item in ypred_for_auc])
    rewards.append(ypred) # seq_length x batch_size
  rewards = np.transpose(np.array(rewards)) # batch_size x seq_length
  return rewards

def split_sentence(input_data, seq_length, padding_value):
  """
  input_data: numpy.array with [batch_size x seq_length]
  """
  # make sure this is 2d array
  assert input_data.ndim == 2
  # Load data 
  # TO-DO better padding with numpy
  datasets = []
  for line in input_data:
      for i in range(1, seq_length+1):
          data = np.pad(line[:i], (0, seq_length-i), 'constant', constant_values=padding_value)
          datasets.append(data)
  return datasets



def split_sentence_file(input_file, output_file, seq_length, padding_value):
  """
  将一个文件内的句子分割成不同长度.
  """
  # Load data
  print("spliting file : ", input_file)
  datasets = []
  with open(input_file) as fin:
    for line in fin:
      line = line.strip().split()
      parse_line = [int(x) for x in line]
      for i in range(1, seq_length+1):
        data = parse_line[:i] + [padding_value] * (seq_length-i)
        datasets.append(data)
  # Output
  with open(output_file, 'w') as fout:
    for data in datasets:
      buffer = ' '.join([str(x) for x in data]) + '\n'
      fout.write(buffer)
  

if __name__ == "__main__":
  SEQ_LENGTH = 32
  input_file = 'tmp/realtrain_cotra.txt'
  output_file = 'tmp/realtrain_cotra.split.txt'
  # split_sentence_file(input_file, output_file, SEQ_LENGTH, 1814)

  ## test split_sentence()
  with open(input_file, 'r') as fin:
    datasets = []
    for i in range(10):
      line = fin.readline().strip().split()
      parse_line = [int(x) for x in line]
      datasets.append(parse_line)
  datasets = np.array(datasets)
  datasets = split_sentence(datasets, 32, 115)
  print(datasets[1])