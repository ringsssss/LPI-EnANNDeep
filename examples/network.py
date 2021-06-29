from DRPLPI_F import *

from deepforest import CascadeForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from itertools import chain
import time

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve,auc
import pandas as pd
import aknn_alg
import numpy as np


def cv3(path):
    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    for times in range(20):
        itime = time.time()
        print("*************%d**************"%(times+1))
        f = getData(np.random.randint(100),path)
        feature = []
        labels = []
        fl = np.array(f)
        [h,l] = fl.shape
        for i in range(h):
            feature.append(fl[i][0:-1])
            labels.append(fl[i][-1])
        nmn = np.array(feature).astype('float64')
        labels = np.array(labels)
        labels = labels.astype('<U1')

        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        for train_index,test_index in kf.split(nmn,labels):
            X_train, X_test = nmn[train_index], nmn[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            nbrs_list = aknn_alg.calc_nbrs_exact(X_test, k=2000)
            aknn_predictions = aknn_alg.predict_nn_rule(nbrs_list, y_test)
            aknn_prediction = list(map(int,aknn_predictions[0]))
            aknn_prediction_prob = list(map(float,aknn_predictions[2]))
            y_train = np.array(list(map(int,y_train)))
            y_test = np.array(list(map(int,y_test)))

            model = Sequential()
            model.add(Dense(554, activation='elu',input_dim = 554))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='elu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='elu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(lr=1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=128,verbose=0)
            score = model.predict(X_test)
            score = list(chain.from_iterable(score))

            forest = CascadeForestClassifier(random_state=1,verbose=0)
            forest.fit(X_train, y_train)
            y_pred_f = forest.predict(X_test)
            allResult = forest.predict_proba(X_test)
            probaResult = np.arange(0, dtype=float)
            for iRes in allResult:
                probaResult = np.append(probaResult, iRes[1]) 

            dnn_pred = []
            score = np.array(score)
            for k in score:
                if k > 0.5:
                    dnn_pred.append(1)
                else:
                    dnn_pred.append(0)
            dnn_pred = np.array(dnn_pred)
            aknn_prediction = np.array(aknn_prediction)
            aknn_prediction_prob = np.array(aknn_prediction_prob)
            mix_res = dnn_pred + aknn_prediction + y_pred_f
            mix_prob = score/3 + aknn_prediction_prob/3 + probaResult/3
            pred = []
            for j in mix_res:
                if j > 1:
                    pred.append(1)
                else:
                    pred.append(0)

            sum_acc += accuracy_score(y_test, pred)
            sum_pre += precision_score(y_test, pred)
            sum_recall += recall_score(y_test, pred)
            sum_f1 += f1_score(y_test, pred)

            fpr, tpr, thresholds = roc_curve(y_test, mix_prob)
            prec, rec, thr = precision_recall_curve(y_test, mix_prob)
            sum_AUC += auc(fpr,tpr)
            sum_AUPR += auc(rec,prec)

        m_acc.append(sum_acc/5)
        m_pre.append(sum_pre/5)
        m_recall.append(sum_recall/5)
        m_F1.append(sum_f1/5)
        m_AUC.append(sum_AUC/5)
        m_AUPR.append(sum_AUPR/5)
        print("accuracy:%.4f+%.4f" % (np.mean(m_acc), np.std(np.array(m_acc))))
        print("precision:%.4f+%.4f" % (np.mean(m_pre), np.std(np.array(m_pre))))
        print("recall:%.4f+%.4f" % (np.mean(m_recall), np.std(np.array(m_recall))))
        print("F1 score:%.4f+%.4f" % (np.mean(m_F1), np.std(np.array(m_F1))))
        print("AUC:%.4f+%.4f" % (np.mean(m_AUC), np.std(np.array(m_AUC))))
        print("AUPR:%.4f+%.4f" % (np.mean(m_AUPR), np.std(np.array(m_AUPR))))
        print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))

    print("precision:%.4f+%.4f"%(np.mean(m_pre),np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f"%(np.mean(m_recall),np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f"%(np.mean(m_acc),np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f"%(np.mean(m_F1),np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f"%(np.mean(m_AUC),np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f"%(np.mean(m_AUPR),np.std(np.array(m_AUPR))))
    

def cv2(path):
    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    name_p = []
    name_R = []
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 0
    RNA_index = 0
    with open(path + 'pro.fasta', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_p.append(line[1:-1])
            else:
                seq = line[:-1]
                protein_seq_dict[name_p[protein_index]] = seq
                protein_index = protein_index + 1
    



    with open(path + 'RNA.fasta', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
            else:
                seq = line[:-1]
                RNA_seq_dict[name_R[RNA_index]] = seq
                RNA_index = RNA_index + 1

    

    df=pd.read_csv(path + "label_T.csv",header=0,index_col=0)

    for times in range(20):
        print("*************%d**************"%(times+1))
        itime = time.time()
        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        for train_index,test_index in kf.split(df.values):

            feature = []
            t_feature = []
            t_labels = []
            s_feature = []
            s_labels = []
            name = []
            X_train = []
            X_test = []
        
            y_train_d, y_test_d = df.values[train_index], df.values[test_index]

            [row,column] = np.where(y_train_d==1)
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = train_index[row[ind]]
            count1_t = 0
            count0_t = 0
            count_t = 0
            count_s = 0
            for i,j in zip(row_index,column):
                RNA_tri_fea = rna_feature_extract(t_feature, RNA_seq_dict[df.columns[j]],path)
                protein_tri_fea = protein_feature_extract(t_feature, protein_seq_dict[str(df.index[i])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(str(df.index[i])+'-'+df.columns[j])
                t_feature.append(temp_f)
                feature.append(temp_f)
                t_labels.append('1')
                count1_t += 1
                count_t += 1

            row = []
            col = []
            [row0,column0] = np.where(y_train_d==0)   
            rand = np.random.RandomState(np.random.randint(100))
            num = rand.randint(row0.shape,size=count1_t)        
            for i in num:
                row.append(row0[i])
                col.append(column0[i])
            
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = train_index[row[ind]]
            
            for i,j in zip(row_index,col):
                RNA_tri_fea = rna_feature_extract(t_feature, RNA_seq_dict[df.columns[j]],path)
                protein_tri_fea = protein_feature_extract(t_feature, protein_seq_dict[str(df.index[i])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(str(df.index[i])+'-'+df.columns[j])
                t_feature.append(temp_f)
                feature.append(temp_f)
                t_labels.append('0')
                count0_t += 1
                count_t += 1


            [row,column] = np.where(y_test_d==1)
            count1_s = 0
            count0_s = 0
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = test_index[row[ind]]
            
            for i,j in zip(row_index,column):
                RNA_tri_fea = rna_feature_extract(s_feature, RNA_seq_dict[df.columns[j]],path)
                protein_tri_fea = protein_feature_extract(s_feature, protein_seq_dict[str(df.index[i])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(str(df.index[i])+'-'+df.columns[j])
                s_feature.append(temp_f)
                feature.append(temp_f)
                s_labels.append('1')
                count1_s += 1
                count_s += 1

            row = []
            col = []
            [row0,column0] = np.where(y_test_d==0)   
            rand = np.random.RandomState(np.random.randint(100))
            num = rand.randint(row0.shape,size=count1_s)        
            for i in num:
                row.append(row0[i])
                col.append(column0[i])
            
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = test_index[row[ind]]
            for i,j in zip(row_index,col):
                RNA_tri_fea = rna_feature_extract(s_feature, RNA_seq_dict[df.columns[j]],path)
                protein_tri_fea = protein_feature_extract(s_feature, protein_seq_dict[str(df.index[i])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(str(df.index[i])+'-'+df.columns[j])
                s_feature.append(temp_f)
                feature.append(temp_f)
                s_labels.append('0')
                count0_s += 1
                count_s += 1


            std = StandardScaler()
            feature = std.fit_transform(feature)

            for i in range(count_t+count_s):
                if i < count_t:
                    X_train.append(feature[i])
                else:
                    X_test.append(feature[i])

            X_train = np.array(X_train).astype('float64')
            X_test = np.array(X_test).astype('float64')
            y_train = np.array(t_labels).astype('<U1')
            y_test = np.array(s_labels).astype('<U1')

            nbrs_list = aknn_alg.calc_nbrs_exact(X_test, k=2000)
            aknn_predictions = aknn_alg.predict_nn_rule(nbrs_list, y_test)
            aknn_prediction = list(map(int,aknn_predictions[0]))
            aknn_prediction_prob = list(map(float,aknn_predictions[2]))
            y_train = np.array(list(map(int,y_train)))
            y_test = np.array(list(map(int,y_test)))

            model = Sequential()
            model.add(Dense(554, activation='elu',input_dim = 554))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='elu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='elu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(lr=1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=128,verbose=0)
            score = model.predict(X_test)
            score = list(chain.from_iterable(score))

            forest = CascadeForestClassifier(random_state=1,verbose=0)
            forest.fit(X_train, y_train)
            y_pred_f = forest.predict(X_test)
            allResult = forest.predict_proba(X_test)
            probaResult = np.arange(0, dtype=float)
            for iRes in allResult:
                probaResult = np.append(probaResult, iRes[1]) 

            

            dnn_pred = []
            score = np.array(score)
            for k in score:
                if k > 0.5:
                    dnn_pred.append(1)
                else:
                    dnn_pred.append(0)
            dnn_pred = np.array(dnn_pred)
            aknn_prediction = np.array(aknn_prediction)
            aknn_prediction_prob = np.array(aknn_prediction_prob)
            mix_res = dnn_pred + aknn_prediction + y_pred_f
            mix_prob = score/3 + aknn_prediction_prob/3 + probaResult/3
            pred = []
            for j in mix_res:
                if j > 1:
                    pred.append(1)
                else:
                    pred.append(0)

            sum_acc += accuracy_score(y_test, pred)
            sum_pre += precision_score(y_test, pred)
            sum_recall += recall_score(y_test, pred)
            sum_f1 += f1_score(y_test, pred)

            fpr, tpr, thresholds = roc_curve(y_test, mix_prob)
            prec, rec, thr = precision_recall_curve(y_test, mix_prob)
            sum_AUC += auc(fpr,tpr)
            sum_AUPR += auc(rec,prec)

        m_acc.append(sum_acc/5)
        m_pre.append(sum_pre/5)
        m_recall.append(sum_recall/5)
        m_F1.append(sum_f1/5)
        m_AUC.append(sum_AUC/5)
        m_AUPR.append(sum_AUPR/5)
        
        print("precision:%.4f+%.4f" % (np.mean(m_pre), np.std(np.array(m_pre))))
        print("recall:%.4f+%.4f" % (np.mean(m_recall), np.std(np.array(m_recall))))
        print("accuracy:%.4f+%.4f" % (np.mean(m_acc), np.std(np.array(m_acc))))
        print("F1 score:%.4f+%.4f" % (np.mean(m_F1), np.std(np.array(m_F1))))
        print("AUC:%.4f+%.4f" % (np.mean(m_AUC), np.std(np.array(m_AUC))))
        print("AUPR:%.4f+%.4f" % (np.mean(m_AUPR), np.std(np.array(m_AUPR))))
        print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))
    
    print("precision:%.4f+%.4f"%(np.mean(m_pre),np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f"%(np.mean(m_recall),np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f"%(np.mean(m_acc),np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f"%(np.mean(m_F1),np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f"%(np.mean(m_AUC),np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f"%(np.mean(m_AUPR),np.std(np.array(m_AUPR))))
    print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))


def cv1(path):
    m_acc = []
    m_pre = []
    m_recall = []
    m_F1 = []
    m_AUC = []
    m_AUPR = []
    name_p = []
    name_R = []
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 0
    RNA_index = 0
    with open(path + 'pro.fasta', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_p.append(line[1:-1])
            else:
                seq = line[:-1]
                protein_seq_dict[name_p[protein_index]] = seq
                protein_index = protein_index + 1
    



    with open(path + 'RNA.fasta', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])
            else:
                seq = line[:-1]
                RNA_seq_dict[name_R[RNA_index]] = seq
                RNA_index = RNA_index + 1

    

    df=pd.read_csv(path + "label.csv",header=0,index_col=0)

    for times in range(20):
        print("*************%d**************"%(times+1))
        itime = time.time()
        sum_acc = 0
        sum_pre = 0
        sum_recall = 0
        sum_f1 = 0
        sum_AUC = 0
        sum_AUPR = 0
        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        for train_index,test_index in kf.split(df.values):

            feature = []
            t_feature = []
            t_labels = []
            s_feature = []
            s_labels = []
            name = []
            X_train = []
            X_test = []
        
            y_train_d, y_test_d = df.values[train_index], df.values[test_index]

            [row,column] = np.where(y_train_d==1)
            count1_t = 0
            count0_t = 0
            count_t = 0
            count_s = 0
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = train_index[row[ind]]
            
            for i,j in zip(row_index,column):
                RNA_tri_fea = rna_feature_extract(t_feature, RNA_seq_dict[df.index[i]],path)
                protein_tri_fea = protein_feature_extract(t_feature, protein_seq_dict[str(df.columns[j])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(df.index[i]+'-'+str(df.columns[j]))
                t_feature.append(temp_f)
                feature.append(temp_f)
                t_labels.append('1')
                count1_t += 1
                count_t += 1

            row = []
            col = []
            [row0,column0] = np.where(y_train_d==0)   
            rand = np.random.RandomState(np.random.randint(100))
            num = rand.randint(row0.shape,size=count1_t)        
            for i in num:
                row.append(row0[i])
                col.append(column0[i])
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = train_index[row[ind]]
            
            for i,j in zip(row_index,col):
                RNA_tri_fea = rna_feature_extract(t_feature, RNA_seq_dict[df.index[i]],path)
                protein_tri_fea = protein_feature_extract(t_feature, protein_seq_dict[str(df.columns[j])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(df.index[i]+'-'+str(df.columns[j]))
                t_feature.append(temp_f)
                feature.append(temp_f)
                t_labels.append('0')
                count0_t += 1
                count_t += 1
            
            [row,column] = np.where(y_test_d==1)
            count1_s = 0
            count0_s = 0
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = test_index[row[ind]]

            for i,j in zip(row_index,column):
                RNA_tri_fea = rna_feature_extract(s_feature, RNA_seq_dict[df.index[i]],path)
                protein_tri_fea = protein_feature_extract(s_feature, protein_seq_dict[str(df.columns[j])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(df.index[i]+'-'+str(df.columns[j]))
                s_feature.append(temp_f)
                feature.append(temp_f)
                s_labels.append('1')
                count1_s += 1
                count_s += 1

            row = []
            col = []
            [row0,column0] = np.where(y_test_d==0)   
            rand = np.random.RandomState(np.random.randint(100))
            num = rand.randint(row0.shape,size=count1_s)        
            for i in num:
                row.append(row0[i])
                col.append(column0[i])
            row_index = np.zeros(len(row),dtype=int)
            for ind in range(len(row)):
                row_index[ind] = test_index[row[ind]]
            
            for i,j in zip(row_index,col):
                RNA_tri_fea = rna_feature_extract(s_feature, RNA_seq_dict[df.index[i]],path)
                protein_tri_fea = protein_feature_extract(s_feature, protein_seq_dict[str(df.columns[j])],path)
                temp_f = list(RNA_tri_fea) + list(protein_tri_fea)
                name.append(df.index[i]+'-'+str(df.columns[j]))
                s_feature.append(temp_f)
                feature.append(temp_f)
                s_labels.append('0')
                count0_s += 1
                count_s += 1
          

            std = StandardScaler()
            feature = std.fit_transform(feature)

            for i in range(count_t+count_s):
                if i < count_t:
                    X_train.append(feature[i])
                else:
                    X_test.append(feature[i])

            X_train = np.array(X_train).astype('float64')
            X_test = np.array(X_test).astype('float64')
            y_train = np.array(t_labels).astype('<U1')
            y_test = np.array(s_labels).astype('<U1')

            nbrs_list = aknn_alg.calc_nbrs_exact(X_test, k=2000)
            aknn_predictions = aknn_alg.predict_nn_rule(nbrs_list, y_test)
            aknn_prediction = list(map(int,aknn_predictions[0]))
            aknn_prediction_prob = list(map(float,aknn_predictions[2]))
            y_train = np.array(list(map(int,y_train)))
            y_test = np.array(list(map(int,y_test)))

            model = Sequential()
            model.add(Dense(554, activation='elu',input_dim = 554))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='elu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='elu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(lr=1e-4),                        
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=128,verbose=0)
            score = model.predict(X_test)
            score = list(chain.from_iterable(score))

            forest = CascadeForestClassifier(random_state=1,verbose=0)
            forest.fit(X_train, y_train)
            y_pred_f = forest.predict(X_test)
            allResult = forest.predict_proba(X_test)
            probaResult = np.arange(0, dtype=float)
            for iRes in allResult:
                probaResult = np.append(probaResult, iRes[1]) 

            

            dnn_pred = []
            score = np.array(score)
            for k in score:
                if k > 0.5:
                    dnn_pred.append(1)
                else:
                    dnn_pred.append(0)
            dnn_pred = np.array(dnn_pred)
            aknn_prediction = np.array(aknn_prediction)
            aknn_prediction_prob = np.array(aknn_prediction_prob)
            mix_res = dnn_pred + aknn_prediction + y_pred_f
            mix_prob = score/3 + aknn_prediction_prob/3 + probaResult/3
            pred = []
            for j in mix_res:
                if j > 1:
                    pred.append(1)
                else:
                    pred.append(0)

            sum_acc += accuracy_score(y_test, pred)
            sum_pre += precision_score(y_test, pred)
            sum_recall += recall_score(y_test, pred)
            sum_f1 += f1_score(y_test, pred)

            fpr, tpr, thresholds = roc_curve(y_test, mix_prob)
            prec, rec, thr = precision_recall_curve(y_test, mix_prob)
            sum_AUC += auc(fpr,tpr)
            sum_AUPR += auc(rec,prec)

        m_acc.append(sum_acc/5)
        m_pre.append(sum_pre/5)
        m_recall.append(sum_recall/5)
        m_F1.append(sum_f1/5)
        m_AUC.append(sum_AUC/5)
        m_AUPR.append(sum_AUPR/5)
        
        print("precision:%.4f+%.4f" % (np.mean(m_pre), np.std(np.array(m_pre))))
        print("recall:%.4f+%.4f" % (np.mean(m_recall), np.std(np.array(m_recall))))
        print("accuracy:%.4f+%.4f" % (np.mean(m_acc), np.std(np.array(m_acc))))
        print("F1 score:%.4f+%.4f" % (np.mean(m_F1), np.std(np.array(m_F1))))
        print("AUC:%.4f+%.4f" % (np.mean(m_AUC), np.std(np.array(m_AUC))))
        print("AUPR:%.4f+%.4f" % (np.mean(m_AUPR), np.std(np.array(m_AUPR))))
        print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))

    
    print("precision:%.4f+%.4f"%(np.mean(m_pre),np.std(np.array(m_pre))))
    print("recall:%.4f+%.4f"%(np.mean(m_recall),np.std(np.array(m_recall))))
    print("accuracy:%.4f+%.4f"%(np.mean(m_acc),np.std(np.array(m_acc))))
    print("F1 score:%.4f+%.4f"%(np.mean(m_F1),np.std(np.array(m_F1))))
    print("AUC:%.4f+%.4f"%(np.mean(m_AUC),np.std(np.array(m_AUC))))
    print("AUPR:%.4f+%.4f"%(np.mean(m_AUPR),np.std(np.array(m_AUPR))))
    print('One time 5-Folds computed. Time: {}m'.format((time.time() - itime)/60))


if __name__ == '__main__':
    print('----------------------cv3------------------')
    cv3('../lncRNA-protein/4/')
    print('----------------------cv2------------------')
    cv2('../lncRNA-protein/4/')
    print('----------------------cv1------------------')
    cv1('../lncRNA-protein/4/')

 



