// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.ArrayList;

class Main
{
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
			LayerLinear.testActivate(false);
			LayerLinear.testBackprop();
			LayerLinear.testUpdateGradient();
			NeuralNet.testCrossValidate();
			NeuralNet.testCrossValidate2();
			NeuralNet.testComputeSumSquaredError();
		} catch (TestFailedException exc) {
			System.out.println(exc);
		}
		//testLearner(new BaselineLearner());
		//testLearner(new RandomForest(50));
		//Matrix.testShuffle();
		
		// Model housing prices in Boston
		
		// Load data into Matrices
		Matrix housingFeatures = new Matrix();
		Matrix housingLabels = new Matrix();
		housingFeatures.loadARFF("data/housing_features.arff");
		housingLabels.loadARFF("data/housing_labels.arff");
		
		// Initialize Layer and NeuralNet
		LayerLinear ols = new LayerLinear(housingFeatures.cols(), housingLabels.cols());
		NeuralNet housingModel = new NeuralNet();
		housingModel.layerCollection.add(ols);
		
		
		// Cross-validate model
		double RMSE = 
			housingModel.crossValidate(housingFeatures, housingLabels, 5, 10);
		System.out.println(RMSE);
		

	}
}
