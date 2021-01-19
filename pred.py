from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary
from fastNLP.models import CNNText
from fastNLP.io import ModelLoader

# get vocab
data_set_loader = CSVLoader(sep='\t')
data_set = data_set_loader._load('data/1/train.tsv')
data_set.rename_field('Phrase', 'raw_words')
data_set.rename_field('Sentiment', 'target')
def get_words(instance):
    ins = instance['raw_words'].split()
    if not ins:
        ins.append('nothing')
    return ins
data_set.apply(get_words, new_field_name='words')
vocab_all = Vocabulary()
vocab_all.from_dataset(data_set, field_name='words')
vocab_all.index_dataset(data_set, field_name='words')
vocab_target = Vocabulary(unknown=None, padding=None)
vocab_target.from_dataset(data_set, field_name='target')
vocab_target.index_dataset(data_set, field_name='target')
#

test_data_loader = CSVLoader(sep='\t')

test_data = test_data_loader._load('data/1/test.tsv')
test_data.rename_field('Phrase', 'raw_words')

test_data.apply(get_words, new_field_name='words')

vocab_all.index_dataset(test_data, field_name='words')

test_data.set_input('words')

EMBED_DIM = 100
model_cnn = CNNText((len(vocab_all),EMBED_DIM), num_classes=len(vocab_target), dropout=0.1)

ModelLoader.load_pytorch(model_cnn, 'save_model/ceshi.pkl')

#pred = model_cnn.predict(torch.LongTensor([test_data[10]['words']]))
def predict(instance):
    pred = model_cnn.predict(torch.LongTensor([instance['words']]))
    pred = vocab_target.to_word(int(pred['pred']))
    return pred
test_data.apply(predict, new_field_name='target')

out_file = open('data/1/sub2021.1.19.csv', 'w')
out_file.write('PhraseId,Sentiment\n')
for ins in test_data:
    line = ins['PhraseId'] + ',' + ins['target'] + '\n'
    out_file.write(line)
out_file.close()
