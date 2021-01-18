from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary

data_set_loader = CSVLoader(sep='\t')

data_set = data_set_loader._load('data/1/train.tsv')
data_set.rename_field('Phrase', 'raw_words')
data_set.rename_field('Sentiment', 'target')
data_set.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')
data_set.drop(lambda ins:ins['raw_words'].strip()=='')

vocab = Vocabulary()
vocab.from_dataset(data_set, field_name='words')
vocab.index_dataset(data_set, field_name='words')

train_data, dev_data = data_set.split(0.015)

# 接着就是按照train这一章做

