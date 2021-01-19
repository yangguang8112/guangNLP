from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary, AccuracyMetric, CrossEntropyLoss, Trainer, Adam
from fastNLP.models import CNNText
import torch.optim as optim
from fastNLP.io import ModelSaver
import torch
from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BertForSequenceClassification

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
#data_set.apply(lambda ins: ins['raw_words'].split(), new_field_name='words')
#data_set.drop(lambda ins:ins['raw_words'].strip()=='')

vocab = Vocabulary()
vocab.from_dataset(data_set, field_name='words')
vocab.index_dataset(data_set, field_name='words')

vocab_target = Vocabulary(unknown=None, padding=None)
vocab_target.from_dataset(data_set, field_name='target')
vocab_target.index_dataset(data_set, field_name='target')

data_set.set_input('words')
data_set.set_target('target')

train_data, dev_data = data_set.split(0.015)

# training
device = 0 if torch.cuda.is_available() else 'cpu'
'''
EMBED_DIM = 100
model = CNNText((len(vocab),EMBED_DIM), num_classes=len(vocab_target), dropout=0.1)
metrics=AccuracyMetric()
loss = CrossEntropyLoss()
optimizer=optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
N_EPOCHS = 10
BATCH_SIZE = 16
trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics,optimizer=optimizer,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, device=device)
trainer.train()
'''
embed = BertEmbedding(vocab, model_dir_or_name='en', include_cls_sep=True)
model = BertForSequenceClassification(embed, len(vocab_target))
trainer = Trainer(train_data, model, optimizer=Adam(model_params=model.parameters(), lr=2e-5),
                    loss=CrossEntropyLoss(),device=device,batch_size=8, dev_data=dev_data,
                    metrics=AccuracyMetric(), n_epochs=2, print_every=1)
trainer.train()

saver = ModelSaver("save_model/bert2021.1.19.pkl")
saver.save_pytorch(model)
