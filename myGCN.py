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
import numpy as np

import tqdm

class MyGCN():
    
    def __init__(
        self,
        x,
        y,
        adjList=None,
        batch_size=None,
        epochs=100,
        dataset='cora', # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
        filter='localpool',
        max_degree=2,  # maximum polynomial degree
        sym_norm=True,  # symmetric (True) vs. left-only (False) normalization
        learning_rate=0.1,
        NB_EPOCH=20,
        PATIENCE=10,  # early stopping patience
    ):
        self.filter=filter
        self.max_degree = max_degree
        self.sym_norm = sym_norm
        self.NB_EPOCH = NB_EPOCH
        self.PATIENCE = PATIENCE
        self.epochs = epochs
        self.learning_rate=learning_rate
        if batch_size is None:
            self.batch_size = x.shape[0]
        else:
            self.batch_size = batch_size
        self.featureNum=x.shape[1]

        if adjList is not None:
            for adjI in range(len(adjList)):
                if type(adjList[adjI])!=sc.sparse.csr_matrix:
                    adjList[adjI]=sc.sparse.csr_matrix(adjList[adjI])
        else:
            adj=np.zeros(x.shape[0],x.shape[0])
            for rowI in range(x.shape[0]):
                for colI in range(x.shape[0]):
                    if np.dot(x[rowI,:],x[colI,:])/np.linalg.norm(x[rowI,:])/np.linalg.norm(x[colI,:])>0.5:
                        adj[rowI,colI]=1
            adjList=[adj]
        self.adjList=adjList
        
        self.support=len(self.adjList)#超图数量
        initIndexList=list(range(self.batch_size))
        initX=x[initIndexList]

        initAdjIndex=np.matrix([(row,col) for row in initIndexList for col in initIndexList]).T.tolist()
        initAdjRowIndex=initAdjIndex[0]
        initAdjColIndex=initAdjIndex[1]
        initAdjList=[sc.sparse.csr_matrix((adjItem[initAdjRowIndex,initAdjColIndex].flatten().tolist()[0],
                                            (initAdjRowIndex,initAdjColIndex)),
                                            shape=[self.batch_size,self.batch_size])
                                for adjItem in self.adjList]
        _, adj_input = self.get_inputs(initAdjList, initX)
        self.myGCNModel=self.build_model(x, y, adj_input)
        
        self.history=[]
        print("training model ...")
        for i in range(self.epochs):
            print("epoch",i)
            for j in range(int(x.shape[0]/self.batch_size)):
                sampleIndexList=(np.random.random_sample(self.batch_size)*x.shape[0]).astype(np.int64).tolist()
                sampleX=x[sampleIndexList,:]
                sampleAdjIndex=np.matrix([(row,col) for row in sampleIndexList for col in sampleIndexList]).T.tolist()
                sampleAdjRowIndex=sampleAdjIndex[0]
                sampleAdjColIndex=sampleAdjIndex[1]
                sampleGraph=[sc.sparse.csr_matrix((adjItem[sampleAdjRowIndex,sampleAdjColIndex].flatten().tolist()[0],
                                                (initAdjRowIndex,initAdjColIndex)),
                                                shape=[self.batch_size,self.batch_size])
                                    for adjItem in self.adjList]
                sampleX_graph, _ = self.get_inputs(sampleGraph, sampleX)

                sampleY=y[sampleIndexList,:]

                self.history.append(self.myGCNModel.fit(sampleX_graph, sampleY, batch_size=self.batch_size, epochs=1, shuffle=False).history["loss"])

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
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=self.learning_rate))
        
        return model


    def predict(self, x, adjList):
        batch_size=x.shape[0]
        zx=x
        adjList=[sc.sparse.csr_matrix(adjItem) for adjItem in adjList]
        x_graph,_ = self.get_inputs(adjList, zx)
        # print(np.max(x_graph[0][:3]))
        return self.myGCNModel.predict(x_graph, batch_size=batch_size)

def splitWord(cw):
    wi=0
    while wi < len(cw):
        # print(ord(cw[wi]))
        if ord(cw[wi])<97 or ord(cw[wi])>122:
            cw=cw[:wi]+" "+cw[wi]+cw[wi+1:]
            wi+=1
        wi+=1
    return cw

def cosSimGroup(w1,w2G,cvModel):
    w1=[splitWord(w1)]
    w2G=[splitWord(w2) for w2 in w2G]
    v1=cvModel.transform(w1).todense()
    v2=cvModel.transform(w2G).todense()
    # print(v1,v2[:5])
    # print(np.dot(v1,v2.T),np.linalg.norm(v1,ord=2),np.linalg.norm(v2,axis=-1,ord=2))
    return np.dot(v1,v2.T)/np.linalg.norm(v1,ord=2)/np.linalg.norm(v2,axis=-1,ord=2)

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
            adj[rowI]=(np.array(cosSimGroup(corpus[rowI,0],corpus[:,0],myCounter))>0.5).astype(np.int64)
                
    y=CountVectorizer().fit_transform(corpus[:,1]).todense()

    print("training model ..")
    model = MyGCN(x, y, adjList=[adj],epochs=15,filter = 'localpool', learning_rate=0.01, batch_size=3)

    print("evaluating model ..")
    preY=model.predict(x[:4],[adj[:4,:4]])
    
    print(preY)
