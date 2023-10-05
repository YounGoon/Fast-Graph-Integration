import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, roc_curve, auc
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import scipy.io
import time

#%%

def graph_construct(data, sigma = 1, neighbor = 15):
    dist = pairwise_distances(data, metric = 'euclidean')
    W_temp = np.exp(-(dist **2) / sigma)
    # W_temp = np.exp(-np.abs(dist) / sigma)
    
    A = kneighbors_graph(data, neighbor, mode='connectivity', include_self = False)
    A = 0.5 * (A + A.T)
    A = A.toarray()
    W = W_temp * A
    L = laplacian(W, normed = True)
    
    return W, L

def graph_SSL(L, y, mu = 1):
    I = np.eye(L.shape[0])
    f = np.linalg.inv(I + mu*L) @ y
    return f


def labeled_Laplacian(L, labeled):
    L_labeled = L[labeled][:, labeled]
    
    return L_labeled 

def softmax(alpha):
    return np.exp(alpha)/sum(np.exp(alpha))

def SGF(L, y, labeled):
    '''
    L have to be 3-d Array!!
    '''
    L_labeled = []
    num_graph = L.shape[0]
    y = y.copy()[labeled]
    
    for i in range(num_graph):
        L_labeled.append(labeled_Laplacian(L[i], labeled))
    
    L_labeled = np.array(L_labeled)
    
    trm = np.zeros((num_graph, num_graph))
    for i in range(num_graph - 1):
        for j in range(i + 1, num_graph):
            trm[i, j] = np.trace(L_labeled[i] @ L_labeled[j])
            
    trm = trm + trm.T
    
    for i in range(num_graph):
        trm[i, i] = np.trace(L_labeled[i] @ L_labeled[i])
            
    smoothness = np.zeros((num_graph, 1))
    for i in range(num_graph):
        # smoothness[i] = num_node - ((y.T @ L[i] @ y) / num_node)
        smoothness[i] = len(labeled) - (y.T @ L_labeled[i] @ y)
    
    alpha = np.linalg.inv(trm) @ smoothness
    alpha = softmax(alpha)
    return alpha 


def FPC(alpha, L, y):
    
    n_data = len(y)
    n_graph = L.shape[0]
    
    y = y.reshape(-1, 1)
    L_int = np.zeros((n_data, n_data))
    for i in range(n_graph):
        L_int = L_int + alpha[i]*L[i]
        
    eta = np.linalg.solve(np.eye(n_data) + L_int, y)
    regval = y.T @ eta

    g = np.zeros(num_graph)
    for i in range(num_graph):
        g[i] = -eta.T @ L[i] @ eta
        
    return regval, g



def GPC(alpha, W, y, labeled):
    
    n_data = len(y)
    n_graph = W.shape[0]
    
    y = y.reshape(-1, 1)
    
    K_Int = np.zeros((n_data, n_data))
    for i in range(n_graph):
        K_Int = K_Int + alpha[i]*W[i] 
        
    M = alpha[-1]*np.eye(n_data) + K_Int
    M_tr = M[labeled][:, labeled]
    M_tr = 0.001*np.eye(len(labeled)) + M_tr
    
    y_tr_gp = y.copy()[labeled]
    
    eta = np.linalg.solve(M_tr, y_tr_gp)
    
    nll = 0.5*y_tr_gp.T @ eta + 0.5*np.trace(np.log(M_tr))
    nll = nll[0][0]    

    return nll



def em_Algorithm(L, y, alpha_init, b, v, labeled):
    
    y = y.copy().reshape(-1, 1)
    num_graph = L.shape[0]
    num_node = L.shape[1]
    
    b_y = b[0]
    b_bias = b[1]
    b_net = b[2]
    
    L_int = np.zeros((num_node, num_node))
    
    for i in range(num_graph):
        L_int = L_int + alpha_init[i]*L[i]
    
    G = np.zeros((num_node, num_node))
    G[labeled, labeled] = 1
    I = np.eye(num_node)
    eta = G + (b_bias/b_y) * I + (b_net / b_y) * L_int

    f_hat = np.linalg.inv(eta) @ (G @ y)
    
    alpha = np.zeros(num_graph)
    for i in range(num_graph):
        alpha[i] = (v + num_node) / (v + b_net*(f_hat.T @ L[i] @ f_hat))

    alpha = softmax(alpha)
    return alpha, f_hat


def RLPMN(L, y, alpha_init, b, v, labeled, tol, tol_step, max_iter):
    
    f_pre = 0
    alpha_pre = alpha_init
    tol_count = 0
    state = True
    iteration = 0
    
    while state:
        iteration = iteration + 1
        
        if iteration == max_iter:
            state = False
        
        alpha, f_hat = em_Algorithm(L, y, alpha_pre, b, v, labeled)
        
        if np.abs(f_hat - f_pre).sum() < tol:
            tol_count = tol_count + 1
            if tol_count == tol_step:
                state = False

        alpha_pre = alpha
        f_pre = f_hat
        
    return alpha

def GeneMANIA(alpha, W, y, labeled, unlabeled):
    
    l = len(labeled)
    l_plus = len(y[y == 1])
    l_minus = len(y[y == -1])
    
    num_graph = W.shape[0]
    W_labeled = np.zeros((num_graph, l, l))
    
    for i in range(num_graph):
        W_labeled[i] = W[i][labeled][:, labeled]
    
    y_labeled = y.copy()[labeled]
    
    t = np.zeros((l, 1))
    
    for i in range(l):
        if y_labeled[i] == -1:
            t[i] = l_plus/l
            
        elif y_labeled[i] == 1:
            t[i] = -l_minus/l
    
    T = t @ t.T
    
    W_int = np.zeros((l, l))
    for i in range(num_graph):
        W_int = W_int + alpha[i]*W_labeled[i]
        
    return np.trace((T - W_int).T @ (T - W_int))
            


def graph_integration(L, alpha):
    
    num_graph = L.shape[0]
    num_node = L.shape[1]
    
    L_integration = np.zeros((num_node, num_node))
    for i in range(num_graph):
        L_integration = L_integration + alpha[i]*L[i]
        
    return L_integration
        
#%%

title = 'DBLP'

directory = 'C:/Users/napol/Desktop/EXP_SGF/data/'

mat_file_name = 'DBLP_mod_v2.mat'
mat_file = scipy.io.loadmat(directory + mat_file_name)

W_total_temp = mat_file['W_Total']
W_total = []

for i in range(W_total_temp.shape[1]):
    W_total.append(W_total_temp[0][i])
    
W_total = np.array(W_total)
    
L_total_temp = mat_file['L_Total']

L_total = []

for i in range(L_total_temp.shape[1]):
    L_total.append(L_total_temp[0][i])

L_total = np.array(L_total)

label = mat_file['y']
num_graph = L_total.shape[0]
num_class = label.shape[1]
# num_class = 1

del mat_file
del W_total_temp
del L_total_temp


#%%

rep_num = 30
ratio = 0.2

proposed_AUC = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
single_AUC = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
fpc_AUC = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
gpc_AUC = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
rlpmn_AUC = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
genem_AUC = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))

proposed_Time = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
fpc_Time = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
gpc_Time = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
rlpmn_Time = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))
genem_Time = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_class))

alpha_proposed_list = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_graph))
alpha_fpc_list = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_graph))
alpha_gpc_list = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_graph))
alpha_rlpmn_list = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_graph))
alpha_genem_list = pd.DataFrame(index = range(1, rep_num + 1), columns = range(num_graph))



for j in range(num_class):
    
    target = label[:, j]
    
    for i in range(rep_num):
        
        print("Class: {}   /   Interation: {}".format(j, i + 1))
        
        y = target.copy()
        labeled, unlabeled = train_test_split(range(len(y)), train_size = ratio,
                                              shuffle = True, stratify = y)
        y[unlabeled] = 0
        y_real = target[unlabeled]
        
        
        # Proposed Method
        print("Proposed Method Start!  Class: {} / Interation: {}".format(j, i + 1))
        start = time.time()
        alpha_proposed = SGF(L_total, y, labeled)
        alpha_proposed = softmax(alpha_proposed)
        proposed_Time.iloc[i, j] = time.time() - start
        alpha_proposed_list.iloc[i, :] = alpha_proposed.reshape(1, -1)
        L_proposed = graph_integration(L_total, alpha_proposed)
        f = graph_SSL(L_proposed, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        proposed_AUC.iloc[i, j] = auc(fpr, tpr)
        
        
        # Single Graph
        print("Single Graph Start!  Class: {} / Interation: {}".format(j, i + 1))
        best_single_AUC = 0
        for k in range(num_graph):
            f = graph_SSL(L_total[k], y, mu = 1)
            y_pred = f[unlabeled]
            fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
            temp_AUC = auc(fpr, tpr)
            if temp_AUC > best_single_AUC:
                best_single_AUC = temp_AUC
        single_AUC.iloc[i, j] = best_single_AUC
        
        
        # Fast Protein Classification
        print("Fast Protein Classification Start!  Class: {} / Interation: {}".format(j, i + 1))
        start = time.time()
        alpha_fpc_0 = (1/num_graph) * np.ones(num_graph)
        st = scipy.optimize.LinearConstraint(np.ones(num_graph).reshape(1, -1), lb = 1, ub = 1)
        alpha_fpc = minimize(FPC, alpha_fpc_0, (L_total, y), bounds = ((0, 1),) * num_graph, constraints = st, jac = True)
        alpha_fpc = alpha_fpc.x
        alpha_fpc = softmax(alpha_fpc)
        fpc_Time.iloc[i,j] = time.time() - start
        alpha_fpc_list.iloc[i, :] = alpha_fpc.reshape(1, -1)
        L_fpc = graph_integration(L_total, alpha_fpc)
        f = graph_SSL(L_fpc, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        fpc_AUC.iloc[i, j] = auc(fpr, tpr)
        
               
        print("Gaussian Process Classification Start!  Class: {} / Interation: {}".format(j, i + 1))
        start = time.time()
        alpha_gpc_0 = (1/num_graph) * np.ones(num_graph + 1)
        alpha_gpc_0[-1] = 100
        bound = [(0, 1000)]*5 + [(10, 1000)]
        alpha_gpc = minimize(GPC, alpha_gpc_0, (W_total, y, labeled), bounds = bound)
        alpha_gpc = alpha_gpc.x[:-1]
        alpha_gpc = softmax(alpha_gpc)
        gpc_Time.iloc[i,j] = time.time() - start
        alpha_gpc_list.iloc[i, :] = alpha_gpc.reshape(1, -1)
        L_gpc = graph_integration(L_total, alpha_gpc)
        f = graph_SSL(L_gpc, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        gpc_AUC.iloc[i, j] = auc(fpr, tpr)
        
        
        
        print("Robust Label Propagation on Multiple Network Start!  Class: {} / Interation: {}".format(j, i + 1))
        start = time.time()
        beta = np.array([1, 0.5, 1])
        v = num_graph - 1
        tol = 0.001
        tol_step = 5
        max_iter = 100
        alpha_init = (1/num_graph) * np.ones(num_graph)
        alpha_rlpmn = RLPMN(L_total, y, alpha_init, beta, v, labeled, tol, tol_step, max_iter)
        alpha_rlpmn = softmax(alpha_rlpmn)
        rlpmn_Time.iloc[i,j] = time.time() - start
        alpha_rlpmn_list.iloc[i, :] = alpha_rlpmn.reshape(1, -1)
        L_rlpmn = graph_integration(L_total, alpha_rlpmn)
        f = graph_SSL(L_rlpmn, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        rlpmn_AUC.iloc[i, j] = auc(fpr, tpr)
        
        
        print("Gene MANIA Start!  Class: {} / Interation: {}".format(j, i + 1))
        start = time.time()
        alpha_genem_0 = (1/num_graph) * np.ones(num_graph)
        bound = ((0, np.inf),) * num_graph
        st = scipy.optimize.LinearConstraint(np.ones(num_graph).reshape(1, -1), lb = 1, ub = 1)
        alpha_genem = minimize(GeneMANIA, alpha_genem_0, (W_total, y, labeled, unlabeled), constraints = st, bounds = bound)
        alpha_genem = alpha_genem.x
        alpha_genem = softmax(alpha_genem)
        genem_Time.iloc[i,j] = time.time() - start
        alpha_genem_list.iloc[i, :] = alpha_genem.reshape(1, -1)
        L_genem = graph_integration(L_total, alpha_genem)
        f = graph_SSL(L_genem, y, mu = 1)
        y_pred = f[unlabeled]
        fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label=1)
        genem_AUC.iloc[i, j] = auc(fpr, tpr)
      
        



#%% save results

directory = 'C:/Users/napol/Desktop/EXP_SGF/result/{}/'.format(title)


proposed_AUC.to_excel(directory + '{}_proposed_AUC({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
single_AUC.to_excel(directory + '{}_single_AUC({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
fpc_AUC.to_excel(directory + '{}_fpc_AUC({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
gpc_AUC.to_excel(directory + '{}_gpc_AUC({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
rlpmn_AUC.to_excel(directory + '{}_rlpmn_AUC({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
genem_AUC.to_excel(directory + '{}_genem_AUC({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))

proposed_Time.to_excel(directory + '{}_proposed_Time({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
fpc_Time.to_excel(directory + '{}_fpc_Time({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
gpc_Time.to_excel(directory + '{}_gpc_Time({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
rlpmn_Time.to_excel(directory + '{}_rlpmn_Time({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
genem_Time.to_excel(directory + '{}_genem_Time({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))

alpha_proposed_list.to_excel(directory + '{}_alpha_proposed({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
alpha_fpc_list.to_excel(directory + '{}_alpha_fpc({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
alpha_gpc_list.to_excel(directory + '{}_alpha_gpc({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
alpha_rlpmn_list.to_excel(directory + '{}_alpha_rlpmn({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))
alpha_genem_list.to_excel(directory + '{}_alpha_genem({}%, {}rep).xlsx'.format(title, ratio*100, rep_num))

