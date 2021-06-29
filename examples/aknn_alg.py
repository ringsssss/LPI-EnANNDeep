import numpy as np, sklearn
import sklearn.metrics



def aknn(nbrs_arr, labels, thresholds, distinct_labels=['0','1']):  

    query_nbrs = labels[nbrs_arr]  

    
    mtr = np.stack([query_nbrs == i for i in distinct_labels]) 
    rngarr = np.arange(len(nbrs_arr))+1 
    fracs_labels = np.cumsum(mtr, axis=1)/rngarr  
    biases = fracs_labels - 1.0/len(distinct_labels)   
    numlabels_predicted = np.sum(biases > thresholds, axis=0)   
    admissible_ndces = np.where(numlabels_predicted > 0)[0]    
    first_admissible_ndx = admissible_ndces[0] if len(admissible_ndces) > 0 else np.argmax(np.abs(biases), axis=1)[0]
    pred_label = distinct_labels[np.argmax(biases[:, first_admissible_ndx])]
    return (pred_label, first_admissible_ndx, fracs_labels[1][first_admissible_ndx])


def predict_nn_rule(nbr_list_sorted, labels, log_complexity=1.0, distinct_labels=['0','1']):
    pred_labels = []
    adaptive_ks = []
    pred_probs  = []
    thresholds = log_complexity/np.sqrt(np.arange(nbr_list_sorted.shape[1])+1)
    distinct_labels = np.unique(labels)
    for i in range(nbr_list_sorted.shape[0]):
        (pred_label, adaptive_k_ndx, pred_prob) = aknn(nbr_list_sorted[i,:], labels, thresholds)
        pred_labels.append(pred_label)
        adaptive_ks.append(adaptive_k_ndx + 1)
        pred_probs.append(pred_prob)
    return (np.array(pred_labels), np.array(adaptive_ks),np.array(pred_probs))


def calc_nbrs_exact(raw_data, k=1000):
    a = sklearn.metrics.pairwise_distances(raw_data)   
    nbr_list_sorted = np.argsort(a, axis=1)[:, 1:]     
    return nbr_list_sorted[:, :k]                      
