import random 
import torch
import numpy as np
from itertools import chain
import warnings
import argparse
from transformers import AutoTokenizer
import os 


SEP_TOKEN_ID = 25

def make_example(tokenizer, rand_seq_length, num_repetitions, dict_lookup, token_sep):
  keys_tokens = list(chain.from_iterable(dict_lookup.keys()))
  sequence = [tokenizer.bos_token_id]
  dict_list = list(dict_lookup.items()) ## keys should be tuples, values should be single tokens
  
  for _ in range(num_repetitions):
    rand_seq = random.choices(keys_tokens, k=rand_seq_length)
    token_key, token_value = random.choice(dict_list)
    sequence += rand_seq + list(token_key) + [token_sep] + [token_value]
  return sequence

def make_dataset(size, tokenizer, rand_seq_length, num_repetitions, key_len, size_dict, token_sep):
  valid_token_ids = [i for i in range(tokenizer.vocab_size) if (i != tokenizer.pad_token_id) and (i != token_sep)]
  examples = []
  for _ in range(size):
    dict_lookup = {}
    for _ in range(size_dict):
      random_token_ids = random.choices(valid_token_ids, k=key_len)
      dict_lookup[tuple(random_token_ids)]  = random.choice(valid_token_ids)
    examples.append(make_example(tokenizer, rand_seq_length, num_repetitions, dict_lookup, token_sep))
  return examples

def main(FLAGS):
  tokenizer = AutoTokenizer.from_pretrained("gpt2", device=FLAGS.device)
  examples = make_dataset(FLAGS.size, tokenizer, FLAGS.rand_seq_length, 
                          FLAGS.num_repetitions, FLAGS.key_len, 
                          FLAGS.size_dict, FLAGS.token_sep)
  examples = np.array(examples)
  os.makedirs(FLAGS.save_path, exist_ok=True)
  np.savetxt(
    os.path.join(FLAGS.save_path, 
                 f'examples-size={FLAGS.size}-rand_seq_len={FLAGS.rand_seq_length}-num_reps={FLAGS.num_repetitions}-key_len={FLAGS.key_len}-dic_sz={FLAGS.size_dict}'), 
    examples, fmt='%i')
  

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  parser = argparse.ArgumentParser(
              prog='Lookup Task Generator',
              description='generates dataset that has a random sequence, then some key token(s), then a separator token, then a value token for some number of repetitions')
  parser.add_argument('--size', default=100, type=int, help='size of the dataset')
  parser.add_argument('--num-repetitions', default=100, type=int, help='maximum number of repeated in context examples')
  parser.add_argument('--rand-seq-length', default=3, type=int, help='how many random tokens before the key value pair')
  parser.add_argument('--key-len', default=1, type=int, help='how many tokens in the key')
  parser.add_argument('--size-dict', default=4, type=int, help='how many key value pairs in the dictionary')
  parser.add_argument('--token-sep', default=25, type=int, help='token id of the separator token (default 25 is colon)')
  parser.add_argument('--save-path', default="datasets/lookup_task/", type=str, help='where to save the dataset')
  parser.add_argument("--device", default="cpu", type=str, help="device to run experiment on")
  
  FLAGS = parser.parse_args()
  main(FLAGS)