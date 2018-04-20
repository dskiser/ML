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
	}		
}
