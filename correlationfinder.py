from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.stats.stats import pearsonr
from scipy import linalg as LA
from sklearn import manifold
from sklearn.metrics import euclidean_distances
import numpy as np
import pandas as pd
import random
import csv

#The name of my csv file containing data
data = "nbaAugmentedQuan.csv"

#Read the csv file and store it into file variable:
file = pd.read_csv(data, header = 0)

file = file._get_numeric_data()
numpyArray = file.as_matrix()

origDataFrame = pd.DataFrame(data=numpyArray[0:,0:], columns = ['Number','Team','Age','Height','Weight','College','Country','Draft Year','Draft Round','Draft Number', 'GP','PTS','REB','AST','NetRtg','OREB%','DREB%','USG%','TS%','AST%'])

origDataFrame.drop(origDataFrame.columns[[0,7,8,9,14,15,16,17,18,19]],axis=1,inplace=True)

corrmatrix = [[0.0 for y in range(numpyArray.shape[1])] for x in range(numpyArray.shape[1])]

pcamatrix = [[0.0 for y in range(numpyArray.shape[1])] for x in range (numpyArray.shape[1])]

for x in range(len(corrmatrix)):
	for y in range(len(corrmatrix)):
		corrmatrix[x][y] = pearsonr(numpyArray[:,x],numpyArray[:,y])[0]

for x in range(len(corrmatrix)):
	for y in range(len(corrmatrix)):
		if ( x == y ):
			pcamatrix[x][y] = np.var(numpyArray[:,x])
		else:		
			pcamatrix[x][y] = pearsonr(numpyArray[:,x],numpyArray[:,y])[0]

pcaFrame = pd.DataFrame(data=pcamatrix, columns=['Number','Team','Age','Height','Weight','College','Country','Draft Year','Draft Round','Draft Number', 'GP','PTS','REB','AST','NetRtg','OREB%','DREB%','USG%','TS%','AST%'])

counter1 = 0
counter2 = 0
newPcaMat = [[0.0 for x in range(10)] for y in range(10)]

for x in range(len(corrmatrix)):
	if(x==1 or x==2 or x==3 or x==4 or x==5 or x==6 or x==10 or x==11 or x==12 or x==13):
		for y in range(len(corrmatrix)):
			if(y==1 or y==2 or y==3 or y==4 or y==5 or y==6 or y==10 or y==11 or y==12 or y==13):
				newPcaMat[counter1][counter2] = pcamatrix[x][y]	
				counter2 = counter2 + 1
		counter1 = counter1 + 1
		counter2 = 0

evals, evecs = LA.eig(newPcaMat)

for x in range(len(evals)):
	evals[x] = abs(evals[x])

evalsum = sum(evals)

for x in range(len(evals)):
	evals[x] = float(evals[x])/float(evalsum)

evalFrame = pd.DataFrame(data=evals, columns=["Evals"])

evalFrame.to_csv(path_or_buf='evalMatrix.csv')

pcaFrame.to_csv(path_or_buf='pcaMatrix.csv')

corrSum = [[0.0 for x in range(len(corrmatrix))] for x in range(1)]

for x in range(len(corrmatrix)):
	for y in range(len(corrmatrix[x])):
		corrSum[0][x] += abs(corrmatrix[x][y])

corrsumFrame = pd.DataFrame(data=corrSum, columns=['Number','Team','Age','Height','Weight','College','Country','Draft Year','Draft Round','Draft Number', 'GP','PTS','REB','AST','NetRtg','OREB%','DREB%','USG%','TS%','AST%'])

corrsumFrame.to_csv(path_or_buf='correlationSums.csv')

corrFrame = pd.DataFrame(data=corrmatrix, columns=['Number','Team','Age','Height','Weight','College','Country','Draft Year','Draft Round','Draft Number', 'GP','PTS','REB','AST','NetRtg','OREB%','DREB%','USG%','TS%','AST%'])

correlationMatrixOfficial = corrFrame.as_matrix()

origData = origDataFrame.as_matrix()

correlationMatrixOfficial = abs(1-correlationMatrixOfficial)

mds = manifold.MDS(dissimilarity="precomputed")

attrmds = mds.fit(correlationMatrixOfficial).embedding_

np.savetxt("AttributeMds.csv", attrmds, delimiter=',')

mds = manifold.MDS(dissimilarity="euclidean")

origmds = mds.fit(origData).embedding_

np.savetxt("EuclideanMds.csv", origmds, delimiter=',')

corrFrame.to_csv(path_or_buf='correlationMatrix.csv')

pcaPlotXY = [[0.0 for x in range(2)] for y in range(numpyArray.shape[0])]

attCounter = 0;

for x in range(numpyArray.shape[0]):
	for y in range(numpyArray.shape[1]):
		if (y==1 or y==2 or y==3 or y==4 or y==5 or y==6 or y==10 or y==11 or y==12 or y==13):
			pcaPlotXY[x][0] = pcaPlotXY[x][0] + evecs[0][attCounter]*numpyArray[x,y]
			pcaPlotXY[x][1] = pcaPlotXY[x][1] + evecs[1][attCounter]*numpyArray[x,y]
			attCounter = attCounter + 1
	attCounter = 0

evecflip = [[0.0 for x in range(2)] for y in range(len(evecs[0]))];

for x in range(len(evecs[0])):
	evecflip[x][0] = evecs[0][x];
	evecflip[x][1] = evecs[1][x];

evecFrame = pd.DataFrame(data=evecflip, columns=['X','Y'])
evecFrame.to_csv(path_or_buf='evecXY.csv');

pcaPlotXYFrame = pd.DataFrame(data=pcaPlotXY, columns=['X','Y'])

pcaPlotXYFrame.to_csv(path_or_buf='pcaplotmatrix.csv')
