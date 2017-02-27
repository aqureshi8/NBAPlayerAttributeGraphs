from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random
import csv

#The name of my csv file containing data
data = "nbaData.csv"

#For encoding and decoding categorical data
teamEnc = preprocessing.LabelEncoder() #encoding team
collegeEnc = preprocessing.LabelEncoder() #encoding college attended
countryEnc = preprocessing.LabelEncoder() #encoding country
#heightEnc = preprocessing.LabelEncoder() #encoding height

#Read the csv file and store it into file variable
file = pd.read_csv(data, header = 0) #return type is a dataframe must change to numpy array

#Encode all categorical data to a number
file.Team = teamEnc.fit_transform(file.Team) #team
file.College = collegeEnc.fit_transform(file.College) #college
file.Country = countryEnc.fit_transform(file.Country) #country
#file.Height = heightEnc.fit_transform(file.Height) #height

#relace all Undrafted in columns WAS NOT RECOGNIZED AS NUMERICAL DATA
#CHANGED INITIAL CSV DATE FILE INSTEAD
#---------------------------------------------------------------------
#for Draft Year: changed to 0
#for x, und in enumerate(file['Draft Year']):
#       if und == 'Undrafted':
#               file['Draft Year'][x] = 0000

#for Draft Round: changed to 3
#for x, und in enumerate(file['Draft Round']):
#       if und == 'Undrafted':
#               file['Draft Round'][x] = 3

#for Draft Number: changed to 70
#for x, und in enumerate(file['Draft Number']):
#       if und == 'Undrafted':
#               file['Draft Number'][x] = 70

#for height: change from feet - inches to just feet
#for x, height in enumerate(file['Height']):
#       heightHolder = [int(stringhere) for stringhere in height.split() if stringhere.isdigit()]
#       file['Height'][x] = float(heightHolder[0]) + float(heightHolder[1])/12.0
# END OF FAILED CHANGES IN CSV
#------------------------------------------------------------------

#for NetRtg: Add 90 to get rid of negative values

for x, netrtgval in enumerate(file['NetRtg']):
	file['NetRtg'][x] = int( file['NetRtg'][x] + 90 )

#for year: subtract 1990

for x, year in enumerate(file['Draft Year']):
	file['Draft Year'][x] = int(file['Draft Year'][x] - 1995)

#get all the numeric data and turn it into a numpy array
file = file._get_numeric_data()

file.to_csv(path_or_buf='nbaDataQuan.csv')
#TEST print(file)
numpyArray = file.as_matrix()

#cluster the data using KMeans Clustering
kmeansClust = KMeans(n_clusters=2).fit(numpyArray)

#TEST print(kmeansClust.inertia_)

#create array of labels. count how many datapoints in each cluster
labelArr = kmeansClust.labels_
labelCount = [0,0]

#set all values in labelCount to 0
for x in labelCount:
        x = 0
#TEST print labelCount

#count up label count
for label in labelArr:
        labelCount[label] += 1
#TEST print labelCount

#TEST print sum(labelCount)

avgdifference = [0 for y in range(2)]

#TEST print(avgdifference)

#TEST print(numpyArray[0])

#find avg values for all fields
for x, cluster in enumerate(kmeansClust.labels_):
        avgdifference[cluster] += abs(kmeansClust.cluster_centers_[cluster]-numpyArray[x])

for x, differencearray in enumerate(avgdifference):
        for y, difference in enumerate(differencearray):
                 avgdifference[x][y] = difference/labelCount[x]

#change labelCount to amount of datapoints I want to add from each cluster
for x, count in enumerate(labelCount):
        labelCount[x] = count/8

newdatapoints = [[0.0 for y in range(19)] for x in range(60)] #holder for new datapoints
datacount = 0 #how many new datapoints have been created

#Create new datapoints into newdatapoints 2D array
for x, clusteramt in enumerate(labelCount):

	clusterColleges = []
	clusterCountry = []

	for n, label in enumerate(kmeansClust.labels_):
		if label == x:
			clusterColleges.append(file['College'][n])
			clusterCountry.append(file['Country'][n])
        for y in range(clusteramt):
                for z in range(19):
			if z == 4:
				newdatapoints[datacount][z] = random.choice(clusterColleges)
			elif z == 5:
				newdatapoints[datacount][z] = random.choice(clusterCountry) 
			elif z == 0:
				newdatapoints[datacount][z] = kmeansClust.cluster_centers_[x][z] + ((random.random() * 2) * avgdifference[x][z]-avgdifference[x][z])
			else:
                        	newdatapoints[datacount][z] = kmeansClust.cluster_centers_[x][z] + ((random.random() * 2 + 1) * avgdifference[x][z] - avgdifference[x][z])
                datacount += 1
#TEST           print datacount

#All data are now floats. Must fix categorical values to translate back
for x, player in enumerate(newdatapoints):
	newdatapoints[x][0] = int(newdatapoints[x][0])
	newdatapoints[x][1] = int(newdatapoints[x][1])
	newdatapoints[x][4] = int(newdatapoints[x][4])
	newdatapoints[x][5] = int(newdatapoints[x][5])
	newdatapoints[x][6] = int(newdatapoints[x][6])
	newdatapoints[x][7] = int(newdatapoints[x][7])
	newdatapoints[x][8] = int(newdatapoints[x][8])
	newdatapoints[x][9] = int(newdatapoints[x][9])

#create a dataframe with the new datapoints
newDataPointFrame = pd.DataFrame(data=newdatapoints, columns=['Team','Age','Height','Weight','College','Country','Draft Year','Draft Round','Draft Number', 'GP','PTS','REB','AST','NetRtg','OREB%','DREB%','USG%','TS%','AST%'])

newDataPointFrame.to_csv(path_or_buf='newDataQuan.csv')

newDataPointFrame.Team = teamEnc.inverse_transform(newDataPointFrame.Team)
newDataPointFrame.Country = countryEnc.inverse_transform(newDataPointFrame.Country)
newDataPointFrame.College = collegeEnc.inverse_transform(newDataPointFrame.College)

#print out newDataPointFrame to a csv

newDataPointFrame.to_csv(path_or_buf='newData.csv')
