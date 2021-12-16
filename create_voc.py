import os
import json
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
from tokenizers import BertWordPieceTokenizer
import glob


# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
  clean_text=False,
  handle_chinese_chars=True,
  strip_accents=False,
  lowercase=True,
)

core_path = "path to txt file /.txt"

with open(core_path, 'r') as f:
    content = f.read()
print('removing new line')
content = content.replace('\n', '\s')

with open('m_text_full.txt', 'w') as f:
    f.write(content)


# prepare text files to train vocab on them
files = ["./m_text_full.txt"]

# train BERT tokenizer
tokenizer.train(
  files,
  vocab_size=60000,
  min_frequency=2,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)
# Verify if it will create the folder './bert-it'
tokenizer.save_model('./bert-it', 'bert-it_full')

# create a BERT tokenizer with trained vocab
vocab = './bert-it/bert-it_full-vocab.txt'
tokenizer = BertWordPieceTokenizer(vocab)

# test the tokenizer with some text
encoded = tokenizer.encode('this file is correct with new test start run at 12pm!')
print(encoded.tokens)



