// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;
import java.text.DecimalFormat;

class Main
{
	public static long startTime;
	
	static void test(SupervisedLearner learner, String challenge)
	{
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		learner.train(trainFeatures, trainLabels);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void testLearner(SupervisedLearner learner)
	{
		test(learner, "hep");
		test(learner, "vow");
		test(learner, "soy");
	}

	public static void main(String[] args)
	{
		
		// Run tests
		
		try {
			LayerLinear.testOrdinary_least_squares(8, 2, false);
			//LayerLinear.testActivate(false);
			//NeuralNet.testGradient(false);
			//NeuralNet.testAssgn4(false);
			//NeuralNet.testCrossValidate();
			//NeuralNet.testCrossValidate2();
			//NeuralNet.testComputeSumSquaredError();
			//NeuralNet.testRefineWeights();
			
		} catch (TestFailedException exc) {
			System.out.println(exc);
		}
		
		//testLearner(new BaselineLearner());
		//testLearner(new RandomForest(50));
		//Matrix.testShuffle();
		//Matrix.testsetRow(); 
		
		/// Crane
		System.out.println("Loading data ... ");
		Matrix X = new Matrix();
		Matrix actions = new Matrix();
		X.loadARFF("data/observations.arff");
		actions.loadARFF("data/actions.arff");
		
		/// Scale X
		X.scale(1.0/256.0);
		//System.out.println(X);
		
		/// Create neural net that becomes observation function
		NeuralNet Obs = new NeuralNet(1.0, 1);
		Obs.layerCollection.add(new LayerLinear(4, 12));
		Obs.layerCollection.add(new LayerTanh(12));
		Obs.layerCollection.add(new LayerLinear(12, 12));
		Obs.layerCollection.add(new LayerTanh(12));
		Obs.layerCollection.add(new LayerLinear(12, 3));
		Obs.layerCollection.add(new LayerTanh(3));
		
		Obs.initWeights();
		
		Matrix V = Obs.train_unsupervised(X);
		V.saveARFF("path.arff");
		
		// Create features for transition model
		Matrix Vminus = new Matrix(V.rows()-1, V.cols()+1);
		Vminus.copyBlock(0, 0, V, 0, 0, V.rows()-1, V.cols());
		Vminus.copyBlock(0, 2, actions, 0, 0, Vminus.rows(), 1);
		
		NomCat nc = new NomCat();
		nc.train(Vminus);
		Matrix trans_input = nc.outputTemplate();
		double[] oldfeat, newfeat;
		int rows = Vminus.rows();
		for(int i = 0; i < rows; i++) {
			oldfeat = Vminus.removeRow(0);
			newfeat = new double[trans_input.cols()];
			nc.transform(oldfeat, newfeat);
			trans_input.takeRow(newfeat);
		}
		
		// Create labels for transition model
		Matrix labels = new Matrix(V.rows()-1, V.cols());
		labels.copyBlock(0, 0, V, 1, 0, labels.rows(), labels.cols());
		
		/// Create neural net that becomes transition function
		NeuralNet Trans = new NeuralNet(1.0, 1);
		Trans.layerCollection.add(new LayerLinear(6, 6));
		Trans.layerCollection.add(new LayerTanh(6));
		Trans.layerCollection.add(new LayerLinear(6, 2));
		
		Trans.initWeights();
		
		System.out.println("Training transition model... ");
		for(int i = 0; i < 100; i++) {
			Trans.train(trans_input, labels);
			//double RMSE = Trans.computeRootMeanSquaredError(trans_input, labels);
			//System.out.println(RMSE);
		}
		
		/// Create images
		System.out.println("Creating images... ");
		Matrix raw_testactions = new Matrix();
		raw_testactions.loadARFF("data/testactions.arff");
		NomCat nc2 = new NomCat();
		nc2.train(raw_testactions);
		Matrix testactions = nc2.outputTemplate();
		rows = raw_testactions.rows();
		for(int i = 0; i < rows; i++) {
			oldfeat = raw_testactions.removeRow(0);
			newfeat = new double[testactions.cols()];
			nc2.transform(oldfeat, newfeat);
			testactions.takeRow(newfeat);
		}
		
		Vec current_action, input;
		Vec current_state = V.row(0);
		String name = "frame";
		String tag = ".png";
		String num, filename;
		
		// Create starting image
		num = Integer.toString(0);
		filename = name + num + tag;
		Obs.makeImage(current_state, filename);
		
		for(int i = 0; i < testactions.rows(); i++) {
			current_action = testactions.row(i);
			input = current_state.attach(current_action);
			current_state = Trans.predict(input);
			
			num = Integer.toString(i+1);
			filename = name + num + tag;
			Obs.makeImage(current_state, filename);
		}
		
	}		
}
