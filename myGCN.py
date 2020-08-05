import tensorflow as tf
from utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from graph import GraphConvolution
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import scipy as sc

import tqdm

class MyGCN():
    
    def __init__(
        self,
        x,
        y,
        adjList,
        batch_size=64,
        epochs=100,
        dataset='cora', # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
        filter='localpool',
        max_degree=2,  # maximum polynomial degree
        sym_norm=True,  # symmetric (True) vs. left-only (False) normalization
        NB_EPOCH=20,
        PATIENCE=10,  # early stopping patience
    ):
        self.filter=filter
        self.max_degree = max_degree
        self.sym_norm = sym_norm
        self.NB_EPOCH = NB_EPOCH
        self.PATIENCE = PATIENCE
        self.epochs = epochs
        self.batch_size = x.shape[0]
        self.featureNum=x.shape[1]
        
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
        self.train_mask=train_mask
        for adjI in range(len(adjList)):
            if type(adjList[adjI])!=sc.sparse.csr_matrix:
                adjList[adjI]=sc.sparse.csr_matrix(adjList[adjI])
        self.adjList=adjList
        
        self.support=len(self.adjList)#超图数量

        x_graph, adj_input = self.get_inputs(self.adjList, x)
        
        self.myGCNModel=self.build_model(x, y, adj_input)
        
        print("training model ...")
        self.history=self.myGCNModel.fit(x_graph, y, sample_weight=self.train_mask,
                            batch_size=self.batch_size, epochs=self.epochs, shuffle=False)

    def get_inputs(self,adjList, x):
        if self.filter == 'localpool':
            print('Using local pooling filters...')
            graph=[x]
            for adjI in range(len(adjList)):
                adj_ = preprocess_adj(adjList[adjI], self.sym_norm)
                adj_ = adj_.todense()
                graph.append(adj_)
            adj_input = [Input(batch_shape=(None, None), sparse=False, name='adj_input'+str(adjI)) for adjI in range(len(adjList))]
        elif self.filter == 'chebyshev':
            print('Using Chebyshev polynomial basis filters...')
            L = normalized_laplacian(adj, self.sym_norm)
            L_scaled = rescale_laplacian(L)
            T_k = chebyshev_polynomial(L_scaled, self.max_degree)
            support = self.max_degree + 1
            graph = [x] + T_k
            adj_input = [Input(batch_shape=(None, None), sparse=False, name='adj_input'+str(i)) for i in range(support)]
        else:
            raise Exception('Invalid filter type.')
        # print(graph[0].shape,graph[1].shape)
        return graph, adj_input
    
    def build_model(self,x, y, adj_input):
        fea_input = Input(shape=(x.shape[1],), name='fea_input')
        net = Dropout(0.5)(fea_input)
        net = GraphConvolution(512, self.support, activation='relu',name="cov1")([net] + adj_input)
        net = Dropout(0.4)(net)
        net = GraphConvolution(256, self.support, activation='relu', kernel_regularizer=l2(5e-4),name="cov2")([net] + adj_input)
        net = Dropout(0.3)(net)
        net = GraphConvolution(128, self.support, activation='relu', kernel_regularizer=l2(5e-4),name="cov3")([net] + adj_input)
        net = Dropout(0.2)(net)
        net = GraphConvolution(64, self.support, activation='relu', kernel_regularizer=l2(5e-4),name="cov4")([net] + adj_input)
        net = Dropout(0.1)(net)
        net = Flatten()(net)
        # output = Dense(y.shape[1], activation='softmax')(net)
        output = GraphConvolution(y.shape[1], self.support, activation='softmax')([net] + adj_input)
        
        model = Model(inputs=[fea_input] + adj_input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
        
        return model


    def predict(self, x, adjList):
        batch_size=x.shape[0]
        zx=x
        adjList=[sc.sparse.csr_matrix(adjItem) for adjItem in adjList]
        x_graph,_ = self.get_inputs(adjList, zx)
        # print(np.max(x_graph[0][:3]))
        return self.myGCNModel.predict(x_graph, batch_size=batch_size)
        

if __name__ == '__main__':
    print("restructuring data ...")
    
    corpus=np.array([
        ["friend hug enermy","hug"],
        ["enermy hug friend","hug"],
        ["hug friend enermy","hug"],
        ["friend you love dog","love"],
        ["you friend love dog","love"],
        ["dog love you","love"],
        ["dog love friend","love"]
    ])
    myCounter=CountVectorizer()
    x=myCounter.fit_transform(corpus[:,0]).todense()
    adj=np.zeros([x.shape[0],x.shape[0]])
    for rowI in range(x.shape[0]):
        for colI in range(x.shape[0]):
            if corpus[rowI][1]==corpus[colI][1]:
                adj[rowI,colI]=1
    y=CountVectorizer().fit_transform(corpus[:,1]).todense()

    print("training model ..")
    model = MyGCN(x, y, [adj],epochs=15,filter = 'localpool')

    print("evaluating model ..")
    preY=model.predict(x[:4],[adj[:4,:4]])
    
    print(preY)