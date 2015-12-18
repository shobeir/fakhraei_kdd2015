
# coding: utf-8

# In[ ]:

# The "relations.csv" should already be sorted based on day and time_ms, otherwise use the Linux sort command first:
# sort -k3n -k1n -k2n relations.csv > relations_sorted.csv


# In[ ]:

import os


# In[ ]:

# Files folder
dataFolder = '../data/'
featuresFolder = '../output/features/' 
if not os.path.exists(featuresFolder):
    os.makedirs(featuresFolder)


# In[ ]:

# Creating the epmty bigram features
bigram_dic = {}

for relation1 in range(0, 8):
    for relation2 in range(0, 8):
        bigram_dic[str(relation1),str(relation2)] = 0


# In[ ]:

# Doing one pass on the relations file and counting the bigram features

sequenceFeatures_dic = {}
lineCount = 0
previousSrc = '0'
previousDst = '0'
previousRelation = '0'
with open(dataFolder+'relations.csv','r') as f:
    for line in f:
        lineVal = line.rstrip('\n').split('\t')
        if lineVal[2] != previousSrc:
            if previousSrc != '0':
                sequenceFeatures_dic[previousSrc] = bigram_dic
            previousSrc = lineVal[2]
            previousRelation = '0'
            bigram_dic = dict.fromkeys(bigram_dic, 0)
        if lineVal[3] != previousDst or lineVal[4] != previousRelation:
            previousDst = lineVal[3]
            if previousRelation != '':
                bigram_dic[previousRelation,lineVal[4]] += 1
        previousRelation = lineVal[4]
                
        lineCount += 1
        #if lineCount > 10000:
        #    break
    
# Last user
sequenceFeatures_dic[previousSrc] = bigram_dic


# In[ ]:

# Saving the bigram features on a file

f = open(featuresFolder+'sequence_bigram_features.csv','w')

outputStr = 'userId'

for relation1 in range(0, 8):
    for relation2 in range(0, 8):
        outputStr += ', '+str(relation1)+'_'+str(relation2)
        
f.write(outputStr+'\n')

#for userId,features in sequenceFeatures_dic.iteritems():
for userId in range(1,5607449):
    if str(userId).zfill(7) in sequenceFeatures_dic:
        features = sequenceFeatures_dic[str(userId).zfill(7)]
        outputStr = ''
        outputStr = str(userId).zfill(7)
        for relation1 in range(0, 8):
            for relation2 in range(0, 8):
                outputStr += ', '+str(features[str(relation1),str(relation2)])
        f.write(outputStr+'\n')

f.close()


# In[ ]:



