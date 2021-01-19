from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary
from fastNLP.models import CNNText
from fastNLP import AccuracyMetric
from fastNLP import CrossEntropyLoss
import torch.optim as optim
from fastNLP import Trainer
from fastNLP.io import ModelSaver

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

EMBED_DIM = 100
model_cnn = CNNText((len(vocab),EMBED_DIM), num_classes=len(vocab_target), dropout=0.1)
metrics=AccuracyMetric()
loss = CrossEntropyLoss()
optimizer=optim.RMSprop(model_cnn.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
N_EPOCHS = 10
BATCH_SIZE = 16
device = 0 if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model=model_cnn, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics,optimizer=optimizer,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, device=device)
trainer.train()

saver = ModelSaver("save_model/ceshi.pkl")
saver.save_pytorch(model_cnn)
