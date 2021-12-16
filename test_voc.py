from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./bert-it/bert-it_18h_peq_novo-vocab.txt')


tokenizer('ciao! come va?')  # hi! how are you?

with open('./bert-it/bert-it_18h_peq_novo-vocab.txt', 'r') as fp:
    vocab = fp.read().split('\n')

vocab[2], vocab[13884], vocab[5], \
vocab[2095], vocab[2281], vocab[35], \
    vocab[3]   