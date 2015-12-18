package fakhraei_kdd2015;

import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.core.*
import edu.umd.cs.psl.core.inference.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.Partition
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.groovy.*
import edu.umd.cs.psl.model.argument.ArgumentType
import edu.umd.cs.psl.model.argument.GroundTerm
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.parameters.PositiveWeight
import edu.umd.cs.psl.model.predicate.Predicate
import edu.umd.cs.psl.ui.loading.*
import edu.umd.cs.psl.util.database.*
import edu.umd.cs.psl.model.atom.*
import edu.umd.cs.psl.application.learning.weight.em.HardEM

// Checking the arguments
if (args.length!=5){
	System.out.println "\nUsage: SpamModel [subModel:1,2,3] [totalFolds] [testFold] [validationFold] [dataFolder]";
	System.out.println "Example: SpamModel 1 3 3 2 'data/'";
	System.exit(0)
}
int model = args[0].toInteger()
int numberOfFolds = args[1].toInteger()
int testFold = args[2].toInteger()
int validationFold = args[3].toInteger()
def base_dir = args[4];

System.out.println "\nStarting..."
System.out.println("totalFolds: "+numberOfFolds)
System.out.println("testFold: "+testFold)
System.out.println("validationFold: "+validationFold)
System.out.println("dataFolder: "+base_dir)

// Setting the config file parameters
ConfigManager cm = ConfigManager.getManager();
ConfigBundle bundle_cfg = cm.getBundle("fakhraei_kdd2015");

// Settings the experiments parameters
today = new Date();
double initialWeight = 1;
boolean sq = true;



// Setting up the database
String dbpath = "./psl_db"+today.getDate()+""+today.getHours()+""+today.getMinutes()+""+today.getSeconds();
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Memory, dbpath, true), bundle_cfg);

// Creating the PSL Model
// ======================
PSLModel m = new PSLModel(this,data)

// Defining Predicates
m.add predicate: "spammer",	types: [ArgumentType.UniqueID]
m.add predicate: "prior_credible",	types: [ArgumentType.UniqueID]
m.add predicate: "credible",	types: [ArgumentType.UniqueID]
m.add predicate: "report",	types: [ArgumentType.UniqueID , ArgumentType.UniqueID]

// Adding rules
if (model == 1)
	{
	// Model 1
	m.add rule : report(User2,User1) >> spammer(User1),  weight : initialWeight, squared: sq
	m.add rule : ~spammer(User) ,  weight : initialWeight, squared: sq
	}
else if (model == 2)
	{
	// Model 2
	m.add rule : (credible(User2) & report(User2,User1)) >> spammer(User1),  weight : initialWeight, squared: sq
	m.add rule : prior_credible(User2) >> credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~prior_credible(User2) >> ~credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~spammer(User) ,  weight : initialWeight, squared: sq
	}
else if (model == 3)
	{
	// Model 3
	m.add rule : (credible(User2) & report(User2,User1)) >> spammer(User1),  weight : initialWeight, squared: sq
	m.add rule : (spammer(User1) & report(User2,User1)) >> credible(User2),  weight : initialWeight, squared: sq
	m.add rule : (~spammer(User1) & report(User2,User1)) >> ~credible(User2),  weight : initialWeight, squared: sq
	m.add rule : prior_credible(User2) >> credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~prior_credible(User2) >> ~credible(User2),  weight : initialWeight, squared: sq
	m.add rule : ~spammer(User) ,  weight : initialWeight, squared: sq
	}

// Printing the model
System.out.println m;

// Creating the partition to read the data
Partition read_pt =  new Partition(1);
Partition write_pt = new Partition(2);
Partition labels_pt = new Partition(3);

Partition read_wl_pt =  new Partition(4);
Partition write_wl_pt = new Partition(5);
Partition labels_wl_pt = new Partition(6);


// Reading from file
	System.out.println "Loading spammer ...";

	//Loading the train set
	for (int i=1; i<=numberOfFolds; i++)
	{
		if (i!=testFold){
			insert = data.getInserter(spammer, read_pt)
			InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+i+'.tsv')			
		}
	}	

	//loding the test labels
	insert = data.getInserter(spammer, labels_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+testFold+'.tsv')
	
	//Loading the test set
	insert = data.getInserter(spammer, write_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'spammer_fold_'+testFold+'_nolabel.tsv')


	// Loadining the train set for WeightLearning
	for (int i=1; i<=numberOfFolds; i++)
	{
		if ((i!=testFold)&&(i!=validationFold)){
			insert = data.getInserter(spammer, read_wl_pt)
			InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+i+'.tsv')
		}
	}	

	// Loading the validation set labels
	insert = data.getInserter(spammer, labels_wl_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'spammer_fold_'+validationFold+'.tsv')

	// Loading the validation set 
	insert = data.getInserter(spammer, write_wl_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'spammer_fold_'+testFold+'_nolabel.tsv')
	InserterUtils.loadDelimitedData(insert, base_dir+'spammer_fold_'+validationFold+'_nolabel.tsv')


	System.out.println "Loading prior_credibility ...";

	insert = data.getInserter(prior_credible, read_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_test_fold_'+testFold+'.tsv')

	insert = data.getInserter(credible, write_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_test_fold_'+testFold+'.tsv')


	insert = data.getInserter(prior_credible, read_wl_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_weightlearning_fold_'+testFold+'.tsv')

	insert = data.getInserter(credible, write_wl_pt)
	InserterUtils.loadDelimitedDataTruth(insert, base_dir+'prior_credibility_weightlearning_fold_'+testFold+'.tsv')

		
	System.out.println "Loading reported ...";
	
	insert = data.getInserter(report, read_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'reported.tsv')
	
	insert = data.getInserter(report, read_wl_pt)
	InserterUtils.loadDelimitedData(insert, base_dir+'reported.tsv')

// Setting which predicates are closed
Set <Predicate>closedPredicates = [prior_credible, report];


// Weight Learning
timeNow = new Date();
System.out.println("\nWeight Learning Start: "+timeNow);
System.out.println("-------------------\n");

Database wl_train_db = data.getDatabase(write_wl_pt, closedPredicates, read_wl_pt);
Database wl_labels_db = data.getDatabase(labels_wl_pt, [spammer] as Set);

HardEM wLearn = new HardEM(m, wl_train_db, wl_labels_db, bundle_cfg);
wLearn.learn();

wl_train_db.close();
wl_labels_db.close();

System.out.println m;


// Inference
timeNow = new Date();
System.out.println("Infernece Start: "+timeNow);
System.out.println("-------------------\n");

Database inference_db = data.getDatabase(write_pt, closedPredicates ,read_pt);

MPEInference mpe = new MPEInference(m, inference_db, bundle_cfg);
FullInferenceResult result = mpe.mpeInference();
mpe.close();
mpe.finalize();

inference_db.close();

timeNow = new Date();

System.out.println("End: "+timeNow);
System.out.println("-------------------\n");


System.out.println("Evaluting ...");

def labels_db = data.getDatabase(labels_pt, closedPredicates)
Database predictions_db = data.getDatabase(new Partition(100), write_pt)

def comparator = new SimpleRankingComparator(predictions_db)
comparator.setBaseline(labels_db)

// Choosing what metrics to report
def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC,  RankingScore.AreaROC]
double [] score = new double[metrics.size()]

try {
	for (int i = 0; i < metrics.size(); i++) {
		comparator.setRankingScore(metrics.get(i))
		score[i] = comparator.compare(spammer)
	}

	System.out.println("\nArea under positive-class PR curve: " + score[0])
	System.out.println("Area under negative-class PR curve: " + score[1])
	System.out.println("Area under ROC curve: " + score[2])
}
catch (ArrayIndexOutOfBoundsException e) {
	System.out.println("No evaluation data! Terminating!");
}


