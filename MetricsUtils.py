# -*- coding: utf-8 -*-

'''
dcl = [1,5,10,20,30,40,50,60,70,80,90,100]
lift_df_ya = lift1(df_ya.true_value,df_ya.prob,dcl)
lift_df_ya
'''
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve,auc

def lift1(y_test, y_pred, dcl, isPlot = False):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    order = np.argsort(y_pred)[::-1]
    y_pred = y_pred[order]
    y_test = y_test[order]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    positive_rate = (float(sum(y_test)) / y_test.shape[0])
    #print "auc:\t\t{}".format(auc)
    #print "positive_rate:\t{} ({}/{})\n".format(positive_rate, sum(y_test), y_test.shape[0] )
    #print "\nk\tlift\t\tprecision\ttp/total\tthreshold"
    #print "{}%\t{:10.3s}{:10.3}{:10}{:10.4}".format("k", "lift", "precision", "tp/num_samples", "threshold")
    df = pd.DataFrame(columns=('k', 'lift', 'precision','tp','num_samples','threshold'))
    for i, percent in enumerate(dcl):
        num_samples = int(float(y_pred.shape[0])*percent/100)
        tp = sum( y_test[:num_samples] )
        total = y_pred[:num_samples].shape[0]+0.0000001
        precision = float(tp)/total
        lift = precision / positive_rate
        threshold = y_pred[:num_samples][-1]
        df.loc[i] = [percent, lift, precision, tp, num_samples, threshold]
     
    if isPlot == True:
        import matplotlib.pyplot as plt
        #pd.set_option('display.height', 1000)
        #pd.set_option('display.max_rows', 1300)
        #pd.set_option('display.max_columns', 500)
        #pd.set_option('display.width', 1000)
        #plt.rcParams['figure.figsize']=(21,17)
        plt.figure(1)
        plt.subplot(221)

        #PLOT LIFT
        int_dcl = df.k.astype(int)
        plt.plot( int_dcl , df['lift'], label='Lift',linewidth= 3 )
        #plt.plot( int_dcl , lift_df_ya['lift'], label='Lift Yandex',linewidth= 3 )
        
        plt.xticks(int_dcl)
        plt.plot(int_dcl, df['lift'],'bo',color='r', label='Lift Value')
        for i in zip(int_dcl, df['lift']):
            plt.text(i[0], i[1] + 0.01, str(round(i[1],2)), fontsize=12)

        #plt.plot(int_dcl[0:4], lift_df_ptb['lift'][0:4],'g^',color='r', label='Lift Value PTB')
        #for i in zip(int_dcl[0:4], lift_df_ptb['lift'][0:4]):
        #    plt.text(i[0], i[1] - 0.15, str(round(i[1],2)), fontsize=12)
    
        plt.legend(loc='upper right', fontsize=15)
        plt.legend(frameon=False, fontsize=15)

        plt.title("Decile Lift OOB", fontsize=15) 
        plt.ylabel('Lift', fontsize=15)
        plt.xlabel('Decile', fontsize=15)

        #PLOT ROC
        plt.subplot(222)

        plt.plot([0, 1], [0, 1], 'k--')

        fpr_ptb, tpr_ptb, _ = roc_curve(y_test,y_pred)
        #fpr_be, tpr_be, _ = roc_curve(df_ya.true_value,df_ya.prob)

        plt.plot(fpr_ptb, tpr_ptb, label='PTB',linewidth= 3 )
        #plt.plot(fpr_be, tpr_be, label='Yandex',linewidth= 3 )

        plt.xlabel('False positive rate', fontsize=15)
        plt.ylabel('True positive rate', fontsize=15)
        plt.title('PTB ROC = ' + str(round(roc_auc_score(y_test,y_pred),2))
            , fontsize=15)
        plt.legend(loc='best', fontsize=15)
        print("Current size:", plt.rcParams["figure.figsize"])
    
    return df