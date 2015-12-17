
# coding: utf-8

# In[ ]:

import graphlab as gl
import os
import numpy as np


# In[ ]:

# Files folder
dataFolder = '../data/'
pslFolder = '../output/psl/'
if not os.path.exists(pslFolder):
    os.makedirs(pslFolder)


# In[ ]:

relations_sf = gl.SFrame.read_csv(dataFolder+'relations_sorted.csv', header=False, delimiter='\t')
relations_sf.rename({'X1':'day','X2':'time_ms','X3':'src','X4':'dst','X5':'relation'})


# In[ ]:

usersData_sf = gl.SFrame.read_csv(dataFolder+'usersdata_sorted.csv', header=False, delimiter='\t')
usersData_sf.rename({'X1':'userId','X2':'sex','X3':'timePassedValidation','X4':'ageGroup','X5':'label'})


# In[ ]:

reports_sf = relations_sf[relations_sf['relation']==7]


# In[ ]:

reportedUsers_sf = reports_sf[['dst']].unique()
reportedUsers_sf.rename({'dst':'userId'})


# In[ ]:

reportedUsers_sf = reportedUsers_sf.join(usersData_sf, how='left')


# In[ ]:

positive_sf = reportedUsers_sf.filter_by([1], "label")
negative_sf = reportedUsers_sf.filter_by([0], "label")

numOfFolds = 3
np.random.seed(2015)

positive_sf['fold']=np.random.random_sample(positive_sf.num_rows())
positive_sf['fold']=positive_sf.apply(lambda x: int(x['fold']*numOfFolds)+1)
positive_sf['shuffle'] = np.random.random_sample(positive_sf.num_rows())
positive_sf = positive_sf.sort('shuffle')
positive_sf.remove_column('shuffle')

negative_sf['fold']=np.random.random_sample(negative_sf.num_rows())
negative_sf['fold']=negative_sf.apply(lambda x: int(x['fold']*numOfFolds)+1)
negative_sf['shuffle'] = np.random.random_sample(negative_sf.num_rows())
negative_sf = negative_sf.sort('shuffle')
negative_sf.remove_column('shuffle')

fullDataset_sf = positive_sf
fullDataset_sf = fullDataset_sf.append(negative_sf)


# In[ ]:

# Exporting the SPAMMER predicate
for currentFold in range(1,numOfFolds+1):

    activeFold_sf = fullDataset_sf[fullDataset_sf['fold']==currentFold]
    activeFold_sf = activeFold_sf.sort('userId')
    activeFold_sf = activeFold_sf[['userId','label']]

    fileOut = open(pslFolder+'spammer_fold_'+str(currentFold)+'.tsv','w')
    for row in activeFold_sf:
        fileOut.write( str(row['userId']).zfill(7)+'\t'+str(row['label'])+'\n')
    fileOut.close() 


# In[ ]:

# Exporting the REPORTED predicate
fileOut = open(pslFolder+'reported.tsv','w')
for row in reports_sf:
    fileOut.write( str(row['src']).zfill(7)+'\t'+str(row['dst'])+'\n')
fileOut.close() 


# In[ ]:

# Exporting PRIOR_CREDIBILITY predicate

def writePriorCredibilityToFile(testFold,weightLearningFold):

    # For Test Prediction
    reportedUsersFold_sf = fullDataset_sf[fullDataset_sf['fold']!=testFold]

    reportingUsersWithLabels_sf = reports_sf.join(reportedUsersFold_sf[['userId','label']], on={'dst':'userId'}, how='inner')[['src','label']]
    reportingUsersWithLabelsCount_sf = reportingUsersWithLabels_sf.groupby(key_columns=['src','label'], operations={'labelReported': gl.aggregate.COUNT()})

    reportingUsers_sf = reportingUsersWithLabels_sf.groupby(key_columns=['src'], operations={'totalReported': gl.aggregate.COUNT()})
    reportingUsers_sf = reportingUsers_sf.join(reportingUsersWithLabelsCount_sf)
    reportingUsers_sf['correctlyReported'] = reportingUsers_sf.apply(lambda x: float(x['labelReported'])/float(x['totalReported']))
    reportingUsers_sf = reportingUsers_sf[reportingUsers_sf['label']==1]
    reportingUsers_sf = reportingUsers_sf.join(reportingUsersWithLabels_sf[['src']].unique(), how='right')
    reportingUsers_sf = reportingUsers_sf.fillna('correctlyReported',0)

    reportingUsers_sf = reportingUsers_sf[['src','correctlyReported']]

    priorCredibility_sf = reports_sf[['src']].unique().join(reportingUsers_sf, on={'src':'src'}, how='left').fillna('correctlyReported',0.5)

    fileOut = open(pslFolder+'prior_credibility_test_fold_'+str(testFold)+'.tsv','w')
    for row in priorCredibility_sf:
        fileOut.write( str(row['src']).zfill(7) + '\t' + str(round(row['correctlyReported'],2)) + '\n' )
    fileOut.close() 
    
    # For Weight Learning Prediction
    reportedUsersFoldWL_sf = reportedUsersFold_sf[reportedUsersFold_sf['fold']!=weightLearningFold]

    reportingUsersWithLabelsWL_sf = reports_sf.join(reportedUsersFoldWL_sf[['userId','label']], on={'dst':'userId'}, how='inner')[['src','label']]
    reportingUsersWithLabelsCountWL_sf = reportingUsersWithLabelsWL_sf.groupby(key_columns=['src','label'], operations={'labelReported': gl.aggregate.COUNT()})

    reportingUsersWL_sf = reportingUsersWithLabelsWL_sf.groupby(key_columns=['src'], operations={'totalReported': gl.aggregate.COUNT()})
    reportingUsersWL_sf = reportingUsersWL_sf.join(reportingUsersWithLabelsCountWL_sf)
    reportingUsersWL_sf['correctlyReported'] = reportingUsersWL_sf.apply(lambda x: float(x['labelReported'])/float(x['totalReported']))
    reportingUsersWL_sf = reportingUsersWL_sf[reportingUsersWL_sf['label']==1]
    reportingUsersWL_sf = reportingUsersWL_sf.join(reportingUsersWithLabelsWL_sf[['src']].unique(), how='right')
    reportingUsersWL_sf = reportingUsersWL_sf.fillna('correctlyReported',0)

    reportingUsersWL_sf = reportingUsersWL_sf[['src','correctlyReported']]

    priorCredibilityWL_sf = reports_sf[['src']].unique().join(reportingUsersWL_sf, on={'src':'src'}, how='left').fillna('correctlyReported',0.5)
    
    fileOut = open(pslFolder+'prior_credibility_weightlearning_fold_'+str(testFold)+'.tsv','w')
    for row in priorCredibilityWL_sf:
        fileOut.write( str(row['src']).zfill(7) + '\t' + str(round(row['correctlyReported'],2)) + '\n' )
    fileOut.close() 


# In[ ]:

for i in range (1, numOfFolds+1):
    if i!=numOfFolds:
        writePriorCredibilityToFile(i,i+1)
    else:
        writePriorCredibilityToFile(i,1)

