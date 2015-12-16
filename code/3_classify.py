
# coding: utf-8

# In[ ]:

import graphlab as gl
import os
import numpy as np
import csv
import time

from sklearn.metrics import precision_recall_curve, roc_curve, auc


# In[ ]:

# Files folder
dataFolder = '../data/'
featuresFolder = '../output/features/' 
predictionsFolder = '../output/predictions/'
if not os.path.exists(predictionsFolder):
    os.makedirs(predictionsFolder)


# In[ ]:

usersData_sf = gl.SFrame.read_csv(dataFolder+'usersdata_sorted.csv', header=False, delimiter='\t')
usersData_sf.rename({'X1':'userId','X2':'sex','X3':'timePassedValidation','X4':'ageGroup','X5':'label'})


# In[ ]:

# Loading the features
graphFeatures_sf = gl.SFrame.read_csv(featuresFolder+'graph_features.csv', header=True)
sequenceFeatures_sf = gl.SFrame.read_csv(featuresFolder+'sequence_bigram_features.csv', header=True)


# In[ ]:

# Selecting the users who have made at least one action
usersData_sf = sequenceFeatures_sf[['userId']].join(usersData_sf, how='left')


# In[ ]:

positive_sf = usersData_sf.filter_by([1], "label")
negative_sf = usersData_sf.filter_by([0], "label")


# In[ ]:

numOfFolds = 10
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


# In[ ]:

# To save the prediction
def saveDict(fn,dict_rap):
    f=open(fn, "w")
    w = csv.writer(f)
    for key, val in dict_rap.items():
        w.writerow([key, val])
    f.close()
     
def readDict(fn):
    f=open(fn,'r')
    dict_rap={}
     
    for key, val in csv.reader(f):
        dict_rap[key]=eval(val)
    f.close()
    return(dict_rap)


# In[ ]:

# Create the fullDataset

# puttin the negative and positive together
fullDataset_sf = positive_sf
fullDataset_sf = fullDataset_sf.append(negative_sf)

# Adding features:
fullDataset_sf = fullDataset_sf.join(graphFeatures_sf, on='userId', how='left')
fullDataset_sf = fullDataset_sf.join(sequenceFeatures_sf, on='userId', how='left')


# In[ ]:

def classify(fullDataset_sf, featureList, featureListName, numOfFolds):
    
    accuracy = {}
    predictions_sf = gl.SFrame()

    for currentFold in range(1,numOfFolds+1):
        
        startTime = time.time()
        
        print str(round(time.time() - startTime,2)) + ": Creating train and test sets - Fold: " + str(currentFold)

        train_sf = fullDataset_sf.filter_by([currentFold],'fold', exclude=True)
        test_sf = fullDataset_sf.filter_by([currentFold],'fold')

        predictionsFold_sf = test_sf[['userId', 'label', 'fold']]

        print str(round(time.time() - startTime,2)) + ": Training the model - Fold: " + str(currentFold)

        model = gl.boosted_trees_classifier.create(train_sf, 
                                            target='label', 
                                            verbose=False,
                                            max_iterations = 3,
                                            class_weights = 'auto',
                                            features = featureList                                        
                                            )

        accuracy[currentFold] = model.evaluate(test_sf)

        print str(round(time.time() - startTime,2))+": Saving the predictions - Fold: " + str(currentFold)
        print "Accuracy: " + str(accuracy[currentFold]['accuracy'])

        predictionsFold_sf['probability'] = model.predict(test_sf, output_type='probability')
        predictionsFold_sf['margin'] = model.predict(test_sf, output_type='margin')
        predictions_sf = predictions_sf.append(predictionsFold_sf)

    print "\n*** Done! Saving all predictions ..."    

    predictions_sf.save(predictionsFolder+'predictions_'+featureListName+'_'+str(numOfFolds)+'_folds.csv', format='csv')    
    saveDict(predictionsFolder+'accuracy_'+featureListName+'_'+str(numOfFolds)+'_folds.txt',accuracy)


# In[ ]:

# Calculating the AUPR and AUC based on the predictions

def evaluate(featureListName, numOfFolds):
    predictions_sf = gl.SFrame.read_csv(predictionsFolder+'predictions_'+featureListName+'_'+str(numOfFolds)+'_folds.csv', verbose=False)
    
    totalAUPR = []
    totalAUROC = []

    for i in range(1,numOfFolds):
        predictionsFold_sf = predictions_sf.filter_by(i,'fold')
        predictionsFold_sf = predictionsFold_sf.sort([('probability', False), ('margin', False)])

        precision, recall, thresholds = precision_recall_curve(y_true=predictionsFold_sf['label'], probas_pred=predictionsFold_sf['probability'], pos_label=1)
        totalAUPR.append(auc(recall, precision))

        fpr, tpr, thresholds = roc_curve(predictionsFold_sf['label'], predictionsFold_sf['probability'])
        totalAUROC.append(auc(fpr, tpr))

    print featureListName
    
    print 'AUPR:'
    print '%0.3f +/- ' %np.mean(totalAUPR) + '%0.3f' %np.std(totalAUPR)

    print 'AUROC:'
    print '%0.3f +/-' %np.mean(totalAUROC) + '%0.3f' %np.std(totalAUROC)


# In[ ]:

# Running the experiments

# graph features
featureList = graphFeatures_sf.column_names()
featureList.remove('userId')
classify(fullDataset_sf, featureList, 'graph_features', numOfFolds)

# sequence features
featureList = sequenceFeatures_sf.column_names()
featureList.remove('userId')
classify(fullDataset_sf, featureList, 'bigram_features', numOfFolds)

# both graph and sequence features
featureList = graphFeatures_sf.column_names()
featureList.remove('userId')
featureList += sequenceFeatures_sf.column_names()
featureList.remove('userId')
classify(fullDataset_sf, featureList, 'graph_and_bigram_features', numOfFolds)

# both graph and sequence and demographics features
featureList = fullDataset_sf.column_names()
featureList.remove('userId')
featureList.remove('fold')
featureList.remove('label')
classify(fullDataset_sf, featureList, 'all_features', numOfFolds)


# In[ ]:

# Computing the results

evaluate('graph_features', numOfFolds)
evaluate('bigram_features', numOfFolds)
evaluate('graph_and_bigram_features', numOfFolds)
evaluate('all_features', numOfFolds)


# In[ ]:



