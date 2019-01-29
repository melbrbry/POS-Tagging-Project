from homework2 import AbstractPOSTaggerTrainer
import numpy as np
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D


class POSTaggerTrainer(AbstractPOSTaggerTrainer):
    
    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir
   
    def load_resources(self):
        self.en_vecmodel = gensim.models.KeyedVectors.load_word2vec_format(self._resource_dir, binary=False)
        pass    
    
    def train(self, training_path):
        
        # choose your preferred embedding mode: 
        # 1 for building the embeddings from scratch through an embedding layer
        # 2 for using pre-traineds embedding and tune them on the training set using an embedding layer
        embed_mode = 1
        
        # choose your preferred model architecture and input features:
        # 1 for using simple LSTM with only word embeddings as inputs
        # 2 for using Bi-LSTM + Convolutional layers using both word, character embeddings and capital features
        # NOTE: the same mode must also be set in the POSTaggerTester
        model_mode = 1
        
        #it_vecmodel = gensim.models.KeyedVectors.load_word2vec_format('/home/elbarbari/t.npy', binary=True)
        #print len(it_vecmodel['la'])
        
        # loading GloVe 100 dimension word embeddings
        features_no = 100
        #en_vecmodel = gensim.models.KeyedVectors.load_word2vec_format('/home/elbarbari/t3.txt', binary=False)
        #it_vecmodel = gensim.models.KeyedVectors.load_word2vec_format('/home/elbarbari/it_word_vec.m', binary=False)
        
                
        
        # building label->index dictionary
        labels = np.array(['ADJ','ADP','ADV','AUX','CONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X'])
        label_to_indx = dict((c,i) for i,c in enumerate(labels))        
        
        # extracting the training dataset from the file
        with open(training_path) as f:
            lines = f.readlines()
    
        words_set = set()
        words,inp,out,tags = [],[],[],[]
        for i in range(len(lines)):
            if lines[i]=='\n':
                inp.append(words)
                out.append(tags)
                tags = []
                words = []
                continue        
            splited = lines[i].split('\t')
            if len(splited)==1:
                continue
            if splited[3]!='_':
                words.append(splited[1])
                words_set.add(splited[1])
                tags.append(splited[3])
        inp = np.array(inp)
        out = np.array(out)
        
        # building embeddings dictionary
        embedding_dic = {}
        for word in words_set:
            try:
                embedding_dic[word] = self.en_vecmodel[word]
            except:
                embedding_dic[word] = np.random.rand(features_no)
        
        # building characters and words dictionaries
        chars = set()
        for word in words_set:
            for char in word:
                chars.add(char)
        
        chars_nb = len(chars)
        words_nb = len(words_set)
        chars_dic = dict((c,i+1) for i,c in enumerate(chars))
        words_dic = dict((c,i+1) for i,c in enumerate(words_set))     
        
        # building words embedding matrix        
        embedding_matrix = np.zeros((words_nb+2, features_no))
        for word, i in words_dic.items():
            try:
                embedding_matrix[i] = self.en_vecmodel[word]
            except KeyError:
                pass        
            
        # preparing the inputs (words, characters, capital features) and the ouput (tag)
        cap_embed_size = cap_size = 4
        max_word_len = 20
        char_inp, cap_inp = [], []
        for i in range(len(inp)):
            char_sent, cap_sent = [], []
            for j in range(len(inp[i])):
                # output 1 hot representation       
                z = np.zeros(len(labels))
                if out[i][j]=='CCONJ':
                    out[i][j]='CONJ'
                indx = label_to_indx[out[i][j]]
                z[indx] = 1
                out[i][j] = z
                
                # characters indices representation (for the embedding layer)
                for k in range(max_word_len):
                    if k>=len(inp[i][j]):
                        char_sent.append(0)
                        continue
                    try:
                        char_sent.append(chars_dic[inp[i][j][k]])
                    except KeyError:
                        char_sent.append(chars_nb+1)
                
                # capital features indices representation
                if inp[i][j].islower():
                    cap_sent.append(1)
                elif inp[i][j].isupper():
                    cap_sent.append(2)
                elif inp[i][j][0].isupper():
                    cap_sent.append(3)
                else:
                    cap_sent.append(4)
                
                # word indices representation
                inp[i][j] = words_dic[inp[i][j]]
                
            char_inp.append(char_sent)
            cap_inp.append(cap_sent)
        
        print "OK"
        # padding
        contxt = 159
        char_embd_dim = 30
        x_train = pad_sequences(inp, maxlen=contxt)
        y_train = pad_sequences(out, maxlen=contxt)
        char_inp = pad_sequences(char_inp, maxlen=max_word_len*contxt)
        cap_inp = pad_sequences(cap_inp, maxlen=contxt)
    
        dropout = 0.2
        
        #x_train = x_train[:10,:]
        #y_train = y_train[:10,:,:]
        #char_inp = char_inp[:10,:]
        #cap_inp = cap_inp[:10,:]
        
        def model_mode1():
            word_input = Input(shape=(contxt,))
            if embed_mode==1:
                word_emb = Embedding(words_nb+2, features_no, input_length=contxt)(word_input)
            elif embed_mode==2:
                word_emb = Embedding(words_nb+2, features_no, weights=[embedding_matrix], input_length=contxt)(word_input)
            emb_droput = Dropout(dropout)(word_emb)
            lstm_word  = LSTM(128,inner_init='uniform', forget_bias_init='one',return_sequences=True)(emb_droput)
            lstm_word_d = Dropout(dropout)(lstm_word)
            dense = TimeDistributed(Dense(17))(lstm_word_d)
            out = Activation('softmax')(dense)
            model = Model(input=word_input, output=out)
            model.compile(loss='categorical_crossentropy', optimizer=RMSprop(0.01))
            model.fit(x_train, y_train, batch_size=32, nb_epoch=1, shuffle=True)
            return model
	           
        def model_mode2():
            char_input = Input(shape=(max_word_len*contxt,))
            char_emb = Embedding(chars_nb+2, char_embd_dim, input_length=max_word_len*contxt)(char_input)
            char_cnn = Convolution1D(nb_filter=30,filter_length=3, activation='tanh', border_mode='full') (char_emb) 
            char_max_pooling = MaxPooling1D(pool_length=max_word_len) (char_cnn)        
            
            caps_input = Input(shape=(contxt,))
            caps_emb = Embedding(cap_size+1, cap_embed_size, input_length=None)(caps_input)
            
            word_input = Input(shape=(contxt,))
            if embed_mode==1:
                word_emb = Embedding(words_nb+2, features_no, input_length=contxt)(word_input)
            elif embed_mode==2:
                word_emb = Embedding(words_nb+2, features_no, weights=[embedding_matrix], input_length=contxt)(word_input)
                
            total_emb = merge([word_emb, caps_emb, char_max_pooling], mode='concat', concat_axis=2)
            
            emb_droput = Dropout(dropout)(total_emb)
            
            bilstm_word  = Bidirectional(LSTM(200,inner_init='uniform', forget_bias_init='one',return_sequences=True))(emb_droput)
            bilstm_word_d = Dropout(dropout)(bilstm_word)
    
            dense = TimeDistributed(Dense(17))(bilstm_word_d)
            out = Activation('softmax')(dense)
            
            model = Model(input=[word_input, caps_input, char_input], output=[out])
            model.compile(loss='categorical_crossentropy', optimizer=RMSprop(0.01))
            
            model.fit([x_train, cap_inp, char_inp], y_train, batch_size=32, nb_epoch=1, shuffle=True)
            return model

        
        if model_mode==1:
            model = model_mode1()
        elif model_mode==2:
            model = model_mode2()
        
        return model
        