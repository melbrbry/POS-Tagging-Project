from homework2 import AbstractPOSTaggerTester
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import gensim
from copy import deepcopy

class POSTaggerTester(AbstractPOSTaggerTester):

    def __init__(self, resource_dir=None):
        self._resource_dir = resource_dir
    
    def load_resources(self):
        self.en_vecmodel = gensim.models.KeyedVectors.load_word2vec_format(self._resource_dir, binary=False)
        pass  
   
    def test(self, model, test_file_path):
        
        # choose your preferred model architecture and input features:
        # 1 for using simple LSTM with only word embeddings as inputs
        # 2 for using Bi-LSTM + Convolutional layers using both word, character embeddings and capital features
        # NOTE: the same mode must also be set in the POSTaggerTrainer
        model_mode = 1
        
        # building label->index dictionaries                    
        labels = np.array(['ADJ','ADP','ADV','AUX','CONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X'])
        label_to_indx = dict((c,i) for i,c in enumerate(labels))
        indx_to_label = dict((i,c) for i,c in enumerate(labels))                  
        
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
        
        # load the testing dataset
        with open(test_file_path) as f:
            lines = f.readlines()
            
        inp, out, tags, words = [],[],[],[]
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
                tags.append(splited[3])
        
                
        row_inp = deepcopy(inp)
        inp = np.array(inp)
        out = np.array(out)
        
        # preparing the test inputs (words, characters, capital features) and the ouput (tag)
        max_word_len = 20
        char_inp, cap_inp = [],[]
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
                try:
                    inp[i][j] = words_dic[inp[i][j]]
                except KeyError:
                    inp[i][j] = words_nb+1

            char_inp.append(char_sent)
            cap_inp.append(cap_sent)
        
        
        # padding
        contxt = 159        
        x_test = pad_sequences(inp, maxlen=contxt)
        y_test = pad_sequences(out, maxlen=contxt)
        char_inp = pad_sequences(char_inp, maxlen=max_word_len*contxt)
        cap_inp = pad_sequences(cap_inp, maxlen=contxt)
                
        if model_mode==1:
            y_predict =  model.predict(x_test)
        elif model_mode==2:
            y_predict =  model.predict([x_test, cap_inp, char_inp])
        
        y_predict = np.array(y_predict)
        # computing precision
        cnt=all=0
        for i in range(len(y_test)):
            for j in range(len(y_test[0])):       
                indx1 = np.where(y_test[i][j]==max(y_test[i][j]))[0][0]
                indx2 = np.where(y_predict[i][j]==max(y_predict[i][j]))[0][0]
                if not np.all(x_test[i][j]==0):
                    all+=1
                    if indx1==indx2:
                        cnt+=1
                            
        precision = 100.0*cnt/all
        
        print "precision:"
        print precision
        
        print "writing"
        f = open('/home/elbarbari/results.txt', 'w')
        
        for i in range(len(row_inp)):
            for j in range(len(row_inp[i])):
                s = row_inp[i][j] + '\t'
                f.write(s)
            f.write('\n')
            for j in range(len(row_inp[i])):
                indx = np.where(y_predict[i][j]==max(y_predict[i][j]))[0][0]
                s = indx_to_label[indx] + '\t'              
            f.write('\n')
        f.close()
        print "finished writing"
######### print confusion matrix        
#        conf_matrix = np.zeros((17,17))
#        for i in range(len(label_predict)):
#            all+=1
#            if label_predict[i]==label_test[i]:
        
        
        
#                cnt+=1
#            else:
#                conf_matrix[label_predict[i]][label_test[i]] += 1        
#        print "Confusion matrix\n", conf_matrix

        
        dic = {'precision': precision, 'recall': precision, 'coverage': 1, 'f1': precision}
        
        return dic