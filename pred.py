from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary
from fastNLP.models import CNNText
from fastNLP.io import ModelLoader
import torch
from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BertForSequenceClassification
from fastNLP.core.utils import _move_model_to_device, _move_dict_value_to_device, _get_model_device

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
'''
EMBED_DIM = 100
model = CNNText((len(vocab_all),EMBED_DIM), num_classes=len(vocab_target), dropout=0.1)
'''
device = 0 if torch.cuda.is_available() else 'cpu'
embed = BertEmbedding(vocab_all, model_dir_or_name='en', include_cls_sep=True)
model = BertForSequenceClassification(embed, len(vocab_target))
ModelLoader.load_pytorch(model, 'save_model/ceshi.pkl')
_move_model_to_device(model, device=device)

#pred = model_cnn.predict(torch.LongTensor([test_data[10]['words']]))
def predict(instance):
    x_batch = torch.LongTensor([instance['words']])
    x_batch = x_batch.to(device=_get_model_device(model))
    pred = model.predict(x_batch)
    pred = vocab_target.to_word(int(pred['pred']))
    return pred

test_data.apply(predict, new_field_name='target')

out_file = open('data/1/sub2021.1.19.csv', 'w')
out_file.write('PhraseId,Sentiment\n')
for ins in test_data:
    line = ins['PhraseId'] + ',' + ins['target'] + '\n'
    out_file.write(line)
out_file.close()
