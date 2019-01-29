from homework2 import AbstractLSTMPOSTagger
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim

class LSTMPOSTagger(AbstractLSTMPOSTagger):
    
    def __init__(self, model, resource_dir=None):
        self._model = model
        self._resource_dir = resource_dir
    
    def load_resources(self):
        self.en_vecmodel = gensim.models.KeyedVectors.load_word2vec_format(self._resource_dir, binary=False)
        pass  

    def get_model(self):
        return self._model

    def predict(self, sentence):
        
        model_mode = 1
        
        labels = np.array(['ADJ','ADP','ADV','AUX','CONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X'])
        indx_to_label = dict((i,c) for i,c in enumerate(labels))
        features_no = 100
        
        # getting the training words to build the same dictionatries built in POSTaggerTrainer
        def get_words(path):
            with open(path) as f:
                lines = f.readlines() 
            words = []
            for i in range(len(lines)):
                splited = lines[i].split('\t')
                if len(splited)==1:
                    continue
                if splited[3]!='_':
                    words.append(splited[1])
            return words
            
        words = []
        for path in ['en-ud-train.conllu']:
            words = words+get_words('/home/elbarbari/'+path)
        
        words_set = set(words)
        chars = set()
        for word in words:
            for char in word:
                chars.add(char)
        chars_nb = len(chars)
        words_nb = len(words_set)
        chars_dic = dict((c,i+1) for i,c in enumerate(chars))
        words_dic = dict((c,i+1) for i,c in enumerate(words_set))        

        inp2 = np.array([sentence])
        
        print inp2.shape
        print inp2
         # preparing the test inputs (words, characters, capital features) and the ouput (tag)
        max_word_len = 20
        char_inp, cap_inp = [],[]
        for i in range(len(inp2)):
            char_sent, cap_sent = [], []
            for j in range(len(inp2[i])):                
                print inp2[i][j]                
                # characters indices representation (for the embedding layer)
                for k in range(max_word_len):
                    if k>=len(inp2[i][j]):
                        char_sent.append(0)
                        continue
                    try:
                        char_sent.append(chars_dic[inp2[i][j][k]])
                    except KeyError:
                        char_sent.append(chars_nb+1)
                
                print inp2[i][j]
                # capital features indices representation
                if inp2[i][j].islower():
                    cap_sent.append(1)
                elif inp2[i][j].isupper():
                    cap_sent.append(2)
                elif inp2[i][j][0].isupper():
                    cap_sent.append(3)
                else:
                    cap_sent.append(4)
                
                # word indices representation
                try:
                    inp2[i][j] = words_dic[inp2[i][j]]
                except KeyError:
                    inp2[i][j] = words_nb+1

            char_inp.append(char_sent)
            cap_inp.append(cap_sent)        
        
        contxt = 159
        x_test = pad_sequences(inp2, maxlen=contxt)
        char_inp = pad_sequences(char_inp, maxlen=max_word_len*contxt)
        cap_inp = pad_sequences(cap_inp, maxlen=contxt)
     
        if model_mode==1:
            y_predict = self._model.predict(x_test)
        elif model_mode==2:
            y_predict = self._model.predict([x_test,cap_inp,char_inp])
        
        label_predict=[]
        for i in range(len(y_predict)):
            for j in range(len(y_predict[0])):
                mx = 0
                indx = 0        
                for k in range(len(y_predict[0][0])):
                    #print y_predict[i][j][k]
                    if y_predict[i][j][k] > mx:
                        mx = y_predict[i][j][k]
                        indx = k
                #print str(indx) + " " + str(indx2)
                label_predict.append(indx_to_label[indx])
        
        return label_predict[:len(sentence)]