from __future__ import print_function
import codecs
import model
import train
import os.path
import numpy as np
import tensorflow as tf
import random
import gzip
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
#import textstat

embedding_dimension = 20
hidden_dimension = 25
#10
seq_len = 100
first_token = 0
EPOCH = 1000
#0.02
switch_rate = 0.001  # supervised to unsupervised training switch
discriminator_steps = 6  # no of discriminator per generator train step
SEED = 88

#partial file
#input_file = 'E:\\Downloads\\Gutenberg\\txt\\outputText.txt'
#Full file
input_file = 'E:\\Downloads\\Gutenberg\\Gutenberg_Full\\gutenbergFull.txt'

discriminator_loss_list=[]
supervised_gen_loss_list=[]
unsupervised_gen_loss_list=[]
supervised_gen_text_list=[]
unsupervised_gen_text_list=[]
    
def tokenizer(input_string):
    return [chr for chr in ' '.join(input_string.split())]

def get_input_data(download=not os.path.exists(input_file)):
    tokenized_input = []
    is_zipped = False
    try:
        open(input_file).read(2)
    except UnicodeDecodeError:
        is_zipped = True
    with gzip.open(input_file) if is_zipped else codecs.open(input_file, 'r', 'utf-8',errors='disregard') as f:
        for line in f:
            line = line if not is_zipped else line.decode('utf-8')        
            tokenized_input.extend(tokenizer(line.strip().lower()))
            tokenized_input.append(' ')
            if len(tokenized_input) > 10000 * seq_len:  # sufficient data
                break

    return tokenized_input


class novelGRU(model.GRU):

    def discriminator_optimiser(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # disregard learning rate

    def generator_optimiser(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # disregard learning rate


def generate_output_text(number_of_embeddings):
    return novelGRU(
        number_of_embeddings, embedding_dimension, hidden_dimension,
        seq_len, first_token)


def generate_random_sequence(tokenized_input, word2idx):
    """random sequence generator"""
    start_idx = random.randint(0, len(tokenized_input) - seq_len)
    return [word2idx[tok] for tok in tokenized_input[start_idx:start_idx + seq_len]]


def check_sequence(three_grams, seq):
    """check 3-grams in text"""
    for i in range(len(seq) - 3):
        if tuple(seq[i:i + 3]) not in three_grams:
            return False
    return True


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tokenized_input = get_input_data()
    assert first_token == 0
    words = ['_BEGIN'] + list(set(tokenized_input))
    word2idx = dict((word, i) for i, word in enumerate(words))
    word_count = len(words)
    three_grams = dict((tuple(word2idx[w] for w in tokenized_input[i:i + 3]), True)
                       for i in range(len(tokenized_input) - 3))
    print('Word count', word_count)
    print('Length of input stream', len(tokenized_input))
    print('unique 3-grams', len(three_grams))
    
    trained_model = generate_output_text(word_count)
    sess = tf.Session()
    sess.run((tf.global_variables_initializer()))
#    #saver.save(sess, 'test_model')

    print('start training')
    discriminator_losses=None 
    supervised_gen_losses=None 
    unsupervised_gen_losses=None
    supervised_generated_text=None
    unsupervised_generated_text=None
    
    for epoch in range(EPOCH):
        print('epoch', epoch)
        supervised_proportion = max(0.0, 1.0 - switch_rate * epoch)
        discriminator_losses,supervised_gen_losses,unsupervised_gen_losses,supervised_generated_text,unsupervised_generated_text=train.train_epoch(
            sess, trained_model, EPOCH,
            supervised_proportion=supervised_proportion,
            generator_steps=1, discriminator_steps=discriminator_steps,
            next_sequence=lambda: generate_random_sequence(tokenized_input, word2idx),
            check_sequence=lambda seq: check_sequence(three_grams, seq),
            words=words)  
        discriminator_loss_list.append(discriminator_losses)
        supervised_gen_loss_list.append(supervised_gen_losses)
        unsupervised_gen_loss_list.append(unsupervised_gen_losses)
        supervised_gen_text_list.append(supervised_generated_text)
        unsupervised_gen_text_list.append(unsupervised_generated_text)
        print('discriminator loss list:',discriminator_loss_list)
        print('generator loss list (supervised):',supervised_gen_loss_list)
        print('generator loss list (unsupervised):',unsupervised_gen_loss_list)
        print('sampled supervised_generated_text',supervised_gen_text_list)
        print('sampled unsupervised_generated_text',unsupervised_gen_text_list)
     
    #plot training loss
    #EPOCH
    epochs=range(0,146)
    d_losses_np=np.array(discriminator_loss_list)
    supervised_g_losses_np=np.array(supervised_gen_loss_list)
    unsupervised_g_losses_np=np.array(unsupervised_gen_loss_list)
   
    plt.plot(epochs,d_losses_np,'r', label='Discriminator loss')
    plt.plot(epochs,supervised_g_losses_np,'g',label='Generator loss(supervised)')
    plt.plot(epochs,unsupervised_g_losses_np,'b',label='Generator loss(unsupervised)')
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig('E:\\NCI\\thesis\\graph\\TrainingLoss.png')
    plt.show()  
    
    #BLEU score metric
    BLEUscore_list_s=[]
    BLEUscore_s=None
    BLEUscore_list_u=[]
    BLEUscore_u=None
    from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
    for i in epochs:
        BLEUscore_s = nltk.translate.bleu_score.sentence_bleu(input_file, supervised_gen_text_list[i] if supervised_gen_text_list[i] is not None else '',weights=(0.50,0.50),smoothing_function=SmoothingFunction().method4)
        BLEUscore_list_s.append(BLEUscore_s)
        BLEUscore_u = nltk.translate.bleu_score.sentence_bleu(input_file, unsupervised_gen_text_list[i] if unsupervised_gen_text_list[i] is not None else '',weights=(0.50,0.50),smoothing_function=SmoothingFunction().method4)
        BLEUscore_list_u.append(BLEUscore_u)
        
    print('BLEU score',BLEUscore_list_s)
    print('BLEU score',BLEUscore_list_u)
    
    BLEUscore_list_s_np=np.array(BLEUscore_list_s)
    BLEUscore_list_u_np=np.array(BLEUscore_list_u)
        
    plt.plot(epochs,BLEUscore_list_s_np,'r',label='supervised generator output')
    plt.plot(epochs,BLEUscore_list_u_np,'g',label='unsupervised generator output')
    plt.title('BLEU score v/s epochs')
    plt.ylabel('BLEU score')
    plt.xlabel('epochs')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.4), shadow=True, ncol=1)
    plt.savefig('E:\\NCI\\thesis\\graph\\BLEUScore.png')
    plt.show()  
        
if __name__ == '__main__':
    main()