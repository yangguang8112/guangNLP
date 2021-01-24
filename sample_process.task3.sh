wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
awk -F"\t" '{print $1"\t"$6"\t"$7}' snli_1.0_train.txt
