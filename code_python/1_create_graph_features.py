
# coding: utf-8

# In[ ]:

import graphlab as gl
import os
import time


# In[ ]:

# Files folder
dataFolder = '../data/'
featuresFolder = '../output/features/' 
if not os.path.exists(featuresFolder):
    os.makedirs(featuresFolder)


# In[ ]:

relations_sf = gl.SFrame.read_csv(dataFolder+'relations.csv', header=False, delimiter='\t')
relations_sf.rename({'X1':'day','X2':'time_ms','X3':'src','X4':'dst','X5':'relation'})


# In[ ]:

usersData_sf = gl.SFrame.read_csv(dataFolder+'usersdata.csv', header=False, delimiter='\t')
usersData_sf.rename({'X1':'userId','X2':'sex','X3':'timePassedValidation','X4':'ageGroup','X5':'label'})


# In[ ]:

# Creating graph features for one relation
def ComputeFeature(activeRelation_sq, activeRelation_sf, usersData_sf, r_id, f_log, num_rows):
    
    outputStr = "Relation_"+r_id+", "+num_rows+"\n"
    
    # The user ids to add features to
    dataFeatures_sf = usersData_sf[['userId']]
    
    # PageRank Feature
    print "PageRank ..."
    timePoint = time.time()
    data_m = gl.pagerank.create(activeRelation_sq, verbose=False)
    dataTempFeature_sf = data_m['pagerank']
    outputStr += "Pagerank, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf.rename({'__id':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')
    dataFeatures_sf.remove_column('delta')
    
    # Triangle Count Feature
    print "Triangle Count ..."
    timePoint = time.time()
    data_m = gl.triangle_counting.create(activeRelation_sq, verbose=False)
    dataTempFeature_sf = data_m['triangle_count']
    outputStr += "Triangle_Count, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf.rename({'__id':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')

    # k-core Feature
    print "k-core ..."
    timePoint = time.time()
    data_m = gl.kcore.create(activeRelation_sq, verbose=False)
    dataTempFeature_sf = data_m['core_id']
    outputStr += "k-core, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf.rename({'__id':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')

    # Graph Coloring Feature
    print "Graph Coloring ..."        
    timePoint = time.time()
    data_m = gl.graph_coloring.create(activeRelation_sq, verbose=False)
    dataTempFeature_sf = data_m['color_id']
    outputStr += "Graph_Coloring, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf.rename({'__id':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')

    # Connected Components Feature
    print "Connected Components ..."
    timePoint = time.time()
    data_m = gl.connected_components.create(activeRelation_sq, verbose=False)
    dataTempFeature_sf = data_m['component_id']
    outputStr += "Connected_Components, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf = dataTempFeature_sf.join(data_m['component_size'], on='component_id', how='left')
    dataTempFeature_sf.rename({'__id':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')
    dataFeatures_sf.rename({'Count':'component_size'})

    # Out-degree Feature
    print "Out-degree ..."
    timePoint = time.time()
    dataTempFeature_sf = activeRelation_sf.groupby("src", {'out_degree':gl.aggregate.COUNT()})
    outputStr += "out_degree, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf.rename({'src':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')

    # In-degree Feature
    print "In-degree ..."
    timePoint = time.time()
    dataTempFeature_sf = activeRelation_sf.groupby("dst", {'in_degree':gl.aggregate.COUNT()})
    outputStr += "in_degree, "+str(time.time()-timePoint)+"\n"
    dataTempFeature_sf.rename({'dst':'userId'})
    dataFeatures_sf = dataFeatures_sf.join(dataTempFeature_sf, on='userId', how='left')
    
    # Filling the missing values
    dataFeatures_sf = dataFeatures_sf.fillna('pagerank',0)
    dataFeatures_sf = dataFeatures_sf.fillna('triangle_count',0)
    dataFeatures_sf = dataFeatures_sf.fillna('core_id',0)
    dataFeatures_sf = dataFeatures_sf.fillna('color_id',0)
    dataFeatures_sf = dataFeatures_sf.fillna('component_id',0)
    dataFeatures_sf = dataFeatures_sf.fillna('component_size',0)
    dataFeatures_sf = dataFeatures_sf.fillna('out_degree',0)    
    dataFeatures_sf = dataFeatures_sf.fillna('in_degree',0)    
    
    # Rename columns
    dataFeatures_sf.rename({'pagerank':r_id+'_pagerank',
                   'triangle_count':r_id+'_triangle_count',
                   'core_id':r_id+'_core_id',
                   'color_id':r_id+'_color_id',
                   'component_id':r_id+'_component_id',
                   'component_size':r_id+'_component_size',
                   'out_degree':r_id+'_out_degree',
                   'in_degree':r_id+'_in_degree'})
    
    outputStr += " \n"
    f_log.write(outputStr)
    
    return dataFeatures_sf


# In[ ]:

# Dataset with all features
dataFeaturesAll_sf = usersData_sf[['userId']]


# In[ ]:

# log file for timing 
f_log = open(featuresFolder+'log_graph_features.csv','w')


# In[ ]:

for i in range(1,8):
    
    print "\nComputing features for relation "+str(i)
    
    # Select one relation
    activeRelation_sf = relations_sf[relations_sf['relation']==i]
    
    # Creating the SGraph for one relation
    activeRelation_sq = gl.SGraph()
    activeRelation_sq = activeRelation_sq.add_vertices(vertices=usersData_sf, vid_field='userId')
    activeRelation_sq = activeRelation_sq.add_edges(edges=activeRelation_sf, src_field='src', dst_field='dst')
    
    dataFeaturesInner_sf = ComputeFeature(activeRelation_sq, activeRelation_sf, usersData_sf, str(i), f_log, str(activeRelation_sf.num_rows()))
    
    dataFeaturesAll_sf = dataFeaturesAll_sf.join(dataFeaturesInner_sf, on='userId', how='left')    
    
f_log.close()


# In[ ]:

# save the features 
# dataFeaturesAll_sf.save(featuresFolder+'graph_features_sf')
dataFeaturesAll_sf.save(featuresFolder+'graph_features.csv', format='csv')

