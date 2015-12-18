#!/bin/bash
# Use this script to run the experiments.

mvn compile
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out

# Total folds in cross-validation
totalFolds=3

# Submodel 1, 2, or 3 (3 is the complete collective models used in the paper)
subModel=3

# Script for cross-validation for sub-model 1:
for testFold in `seq 1 $totalFolds`; do

    # Setting the validation fold
    if [ $testFold = 1 ]; then
        let "validationFold=totalFolds"
    else
        let "validationFold=testFold-1"
    fi

    # Running the model
    java -Xms2G -Xmx4G -cp ./target/classes:`cat classpath.out` fakhraei_kdd2015.SpamModel $subModel $totalFolds $testFold $validationFold 'data/'
    
    echo "*** Fold $testFold Done! ***"
done
 
 

