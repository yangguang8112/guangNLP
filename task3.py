from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary, AccuracyMetric, CrossEntropyLoss, Trainer, Adam
import torch.optim as optim
from fastNLP.io import ModelSaver
import torch
from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BertForSequenceClassification
from fastNLP.models import ESIM


data_set_loader = CSVLoader(sep='\t')

train_data = data_set_loader._load('data/task3/snli_1.0/my.snli_1.0_train.txt')
dev_data = data_set_loader._load('data/task3/snli_1.0/my.snli_1.0_dev.txt')
test_data = data_set_loader._load('data/task3/snli_1.0/my.snli_1.0_test.txt')

def sample_process(data_set):
    data_set.rename_field('gold_label', 'target')
    data_set.rename_field('sentence1', 'raw_words1')
    data_set.rename_field('sentence2', 'raw_words2')
    data_set.apply(lambda ins: ins['raw_words1'].split(), new_field_name='words1')
    data_set.apply(lambda ins: ins['raw_words2'].split(), new_field_name='words2')
    data_set.apply(lambda ins: len(ins['words1']), new_field_name='seq_len1')
    data_set.apply(lambda ins: len(ins['words2']), new_field_name='seq_len2')
    data_set.apply(lambda ins: ins['words1']+ins['words2'], new_field_name='words')
    return

sample_process(train_data)
sample_process(dev_data)
sample_process(test_data)

vocab = Vocabulary()
vocab.from_dataset(train_data, field_name='words')
vocab.index_dataset(train_data, field_name='words1')
vocab.index_dataset(train_data, field_name='words2')

#
data_set = train_data

vocab_target = Vocabulary(unknown=None, padding=None)
vocab_target.from_dataset(data_set, field_name='target')
vocab_target.index_dataset(data_set, field_name='target')

data_set.set_input('words1','words2', 'seq_len1', 'seq_len2')
data_set.set_target('target')

train_data, dev_data = data_set.split(0.015)


# training
device = 0 if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 100
model_esim = ESIM((len(vocab),EMBED_DIM), num_labels=len(vocab_target), dropout_rate=0.1)
metrics=AccuracyMetric()
loss = CrossEntropyLoss()
optimizer=optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
N_EPOCHS = 10
BATCH_SIZE = 16
trainer = Trainer(model=model_esim, train_data=train_data, dev_data=dev_data, loss=loss, metrics=metrics,optimizer=optimizer,n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, device=device)
trainer.train()



saver = ModelSaver("save_model/bert2021.1.19.pkl")
saver.save_pytorch(model)