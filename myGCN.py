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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import jieba
import pandas as pd

import tqdm
import pickle as pkl

class MyGCN():
    
    def __init__(
        self,
        maxLen=20,
        classNum=2,
        support=1,
        filter='localpool',
        max_degree=2,  # maximum polynomial degree
        sym_norm=True,  # symmetric (True) vs. left-only (False) normalization
        learning_rate=0.1,
        NB_EPOCH=20,
        PATIENCE=10,  # early stopping patience
    ):
        self.classNum=classNum
        self.maxLen=maxLen
        self.filter=filter
        self.max_degree = max_degree
        self.sym_norm = sym_norm
        self.NB_EPOCH = NB_EPOCH
        self.PATIENCE = PATIENCE
        self.learning_rate=learning_rate
        self.support=support
        
        adj_input = self.getAdjInput()
        self.model=self.build_model(adj_input)

    def getAdjInput(self):
        adj_input = [Input(batch_shape=(None, None), sparse=False, name='adj_input'+str(adjI)) for adjI in range(self.support)]
        return adj_input


    def build_model(self,adj_input):
        fea_input = Input(shape=(self.maxLen,), name='fea_input')
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
        output = GraphConvolution(self.classNum, self.support, activation='softmax')([net] + adj_input)
        
        model = Model(inputs=[fea_input] + adj_input, outputs=output)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=self.learning_rate))
        
        return model

    def get_inputs(self,adjList, x):
        if self.filter == 'localpool':
            graph=[np.array(x).astype(float)]
            for adjI in range(len(adjList)):
                adj_ = preprocess_adj(adjList[adjI], self.sym_norm)
                adj_ = adj_.todense()
                graph.append(np.array(adj_).astype(float))
            adj_input = [Input(batch_shape=(None, None), sparse=False, name='adj_input'+str(adjI)) for adjI in range(len(adjList))]
        elif self.filter == 'chebyshev':
            L = normalized_laplacian(adj, self.sym_norm)
            L_scaled = rescale_laplacian(L)
            T_k = chebyshev_polynomial(L_scaled, self.max_degree)
            support = self.max_degree + 1
            graph = [np.array(x).astype(float)] + T_k
            adj_input = [Input(batch_shape=(None, None), sparse=False, name='adj_input'+str(i)) for i in range(support)]
        else:
            raise Exception('Invalid filter type.')
        return graph, adj_input

    def cosSimGroup(self,w1,w2G):
        if isinstance(w1,sc.sparse.csr_matrix):
            w1=w1.todense()
        if isinstance(w2G,sc.sparse.csr_matrix):
            w1=w2G.todense()
        return np.dot(w1,w2G.T)/np.linalg.norm(w1,ord=2)/np.linalg.norm(w2G,axis=-1,ord=2)

    def fit(self,x,y,adjList=None,epochs=15,batch_size=32):
        # x=x.astype(float)
        self.batch_size=min(batch_size,x.shape[0])
        self.history=[]

        if adjList is None:
            adj=np.zeros([x.shape[0],x.shape[0]])
            for rowI in tqdm.tqdm(range(x.shape[0])):
                adj[rowI]=(np.array(self.cosSimGroup(x[rowI,:],x))>0.5).astype(np.int64)
            adjList=[np.matrix(adj)]

        initAdjRowIndex=[row for row in range(self.batch_size) for col in range(self.batch_size)]
        initAdjColIndex=[col for row in range(self.batch_size) for col in range(self.batch_size)]
        print("training model ...")
        for i in range(epochs):
            print("epoch:{}/{}".format(i+1,epochs))
            for j in tqdm.tqdm(range(int(x.shape[0]/self.batch_size))):
                sampleIndexList=(np.random.random_sample(self.batch_size)*x.shape[0]).astype(np.int64).tolist()
                sampleX=x[sampleIndexList,:]
                sampleAdjIndex=np.matrix([(row,col) for row in sampleIndexList for col in sampleIndexList]).T.tolist()
                sampleAdjRowIndex=sampleAdjIndex[0]
                sampleAdjColIndex=sampleAdjIndex[1]

                sampleAdjList=[sc.sparse.csr_matrix((adjItem[sampleAdjRowIndex,sampleAdjColIndex].flatten().tolist()[0],
                                                (initAdjRowIndex,initAdjColIndex)),
                                                shape=[self.batch_size,self.batch_size])
                                    for adjItem in adjList]
                                    
                sampleX_graph, _ = self.get_inputs(sampleAdjList, sampleX)

                sampleY=y[sampleIndexList].astype(float)

                loss=self.model.fit(sampleX_graph, sampleY, batch_size=self.batch_size, epochs=1, shuffle=False, verbose=False).history["loss"][0]
            print("loss:",loss)


    def predict(self, x,adjList=None):

        if adjList is None:
            adj=np.zeros([x.shape[0],x.shape[0]])
            for rowI in tqdm.tqdm(range(x.shape[0])):
                adj[rowI]=(np.array(self.cosSimGroup(x[rowI,:],x))>0.5).astype(np.int64)
            adjList=[np.matrix(adj)]
        
        preYList=[]
        adjSize=self.batch_size
        for batchI in tqdm.tqdm(range(int(x.shape[0]/self.batch_size)+1)):
            initAdjRowIndex=[row for row in range(self.batch_size) for col in range(self.batch_size)]
            initAdjColIndex=[col for row in range(self.batch_size) for col in range(self.batch_size)]
            sampleIndexList=list(range(batchI*self.batch_size,min((batchI+1)*self.batch_size,x.shape[0])))
            if len(sampleIndexList)<self.batch_size and len(sampleIndexList)>0:
                adjIndex=0
                while adjIndex < len(initAdjRowIndex):
                    if (initAdjRowIndex[adjIndex] < len(sampleIndexList) and initAdjColIndex[adjIndex] < len(sampleIndexList))==False:
                        initAdjRowIndex.pop(adjIndex)
                        initAdjColIndex.pop(adjIndex)
                        adjIndex-=1
                    adjIndex+=1
                adjSize=len(sampleIndexList)
            elif len(sampleIndexList)==0:
                break
            sampleAdjIndex=np.matrix([(row,col) for row in sampleIndexList for col in sampleIndexList]).T.tolist()
            sampleAdjRowIndex=sampleAdjIndex[0]
            sampleAdjColIndex=sampleAdjIndex[1]
            sampleX=x[sampleIndexList]
            ss=adjList[0][sampleAdjRowIndex,sampleAdjColIndex].flatten().tolist()[0]
            sampleAdjList=[sc.sparse.csr_matrix((adjItem[sampleAdjRowIndex,sampleAdjColIndex].flatten().tolist()[0],
                                                (initAdjRowIndex,initAdjColIndex)),
                                                shape=[adjSize,adjSize])
                                    for adjItem in adjList]
            x_graph,_ = self.get_inputs(sampleAdjList, sampleX)
            preY=self.model.predict(x_graph, batch_size=self.batch_size)
            preYList+=preY.tolist()
            # print(np.max(x_graph[0][:3]))
        return np.array(preYList)

def splitWord(cw):
    wi=0
    while wi < len(cw):
        # print(ord(cw[wi]))
        if ord(cw[wi])<97 or ord(cw[wi])>122:
            cw=cw[:wi]+" "+cw[wi]+cw[wi+1:]
            wi+=1
        wi+=1
    return cw

if __name__ == '__main__':
    print("restructuring data ...")
    corpusDf=pd.read_csv("data/example.csv").loc[:,["text","class"]]
    corpusDf["text"]=corpusDf["text"].astype(str).apply(lambda row:" ".join(list(jieba.cut(row))))
    corpus=corpusDf.values

    print("building vec/class vectorizer ...")
    xCounter=CountVectorizer(min_df=0, token_pattern='\w+')
    x=xCounter.fit_transform([splitWord(row[0]) for row in corpus[:,0]]).todense()
    maxLen=x.shape[1]

    y=corpus[:,1].astype(int)
    classNum=len(list(set(y.tolist())))

    print("splitting train/test data ...")
    trainX,testX,trainY,testY=train_test_split(x,y,test_size=0.2)

    print("building model ...")
    learning_rate=0.0001
    model = MyGCN(maxLen=maxLen,classNum=classNum, learning_rate=learning_rate)
    model.model.summary()

    print("training model ...")
    model.fit(trainX, trainY,epochs=500, batch_size=256)

    print("evaluating model ...")
    preY=model.predict(testX)
    preYC=np.argmax(preY,axis=-1)
    yC=testY
    print("f1:",f1_score(yC,preYC,average="macro"))


    print("saving model ...")
    with open("model/xCounter.pkl","wb+") as xCounterFile:
        pkl.dump(xCounter,xCounterFile)
    # with open("model/yCounter.pkl","wb+") as yCounterFile:
    #     pkl.dump(yCounter,yCounterFile)
    with open("model/myGCNHp.pkl","wb+") as modelHpFile:
        myGCNHp={
            "maxLen":maxLen,
            "classNum":classNum,
            "learning_rate":learning_rate
        }
        pkl.dump(myGCNHp,modelHpFile)
    model.model.save_weights("model/myGCNModel")
    