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
		} catch (TestFailedException exc) {
			System.out.println(exc);
		}
		//testLearner(new BaselineLearner());
		//testLearner(new RandomForest(50));
		
		// Test NeuralNet
		int inputs = 2;
		int outputs = 1;
		LayerLinear testlayer = new LayerLinear(inputs, outputs);
		ArrayList<Layer> layerlist = new ArrayList<Layer>();
		layerlist.add(testlayer);
		NeuralNet testnet = new NeuralNet(
		double[] x_values = { 1 , 3 };
		Vec x = new Vec(x_values);
		Matrix X  = new Matrix(3, 2);
		for(int i=0; i<3; i++) {
			x_values[0] += i;
			X.takeRow(x_values);
		}
		Matrix Y = new Matrix(3, 1);
		for(int i=0; i<3; i++) {Y.get(i, 0) = i;}
		testnet.train(X, Y);
		System.out.println(testnet.layerCollection(0).activation);
		
	}
}
