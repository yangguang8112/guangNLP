from fastNLP.io.loader import CSVLoader
from fastNLP import Vocabulary, AccuracyMetric, CrossEntropyLoss, Trainer, Adam



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
    return

sample_process(train_data)
sample_process(dev_data)
sample_process(test_data)

