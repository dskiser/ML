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
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels, false);
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
			NeuralNet.testGradient();
			//NeuralNet.testCrossValidate();
			//NeuralNet.testCrossValidate2();
			NeuralNet.testComputeSumSquaredError();
			NeuralNet.testRefineWeights();
		} catch (TestFailedException exc) {
			System.out.println(exc);
		}
		//testLearner(new BaselineLearner());
		//testLearner(new RandomForest(50));
		//Matrix.testShuffle();
		
		/// Predict handwritten digits
		
		/// Load data into Matrices
		//System.out.println("Loading data . . . ");
		
		// Training data
		Matrix trainX = new Matrix();
		Matrix trainY = new Matrix();
		trainX.loadARFF("data/train_feat.arff");
		trainY.loadARFF("data/train_lab.arff");
		
		// Convert trainY to one-hot representation
		Matrix onehotTrain = new Matrix(0, 10);
		for(int i = 0; i < trainY.rows(); i++) {
			double[] y_row = new double[10];
			int index = (int) trainY.get(i, 0);
			y_row[index] = 1.0;
			onehotTrain.takeRow(y_row);
		}
		
		// Test data
		Matrix testX = new Matrix();
		Matrix testY = new Matrix();
		testX.loadARFF("data/test_feat.arff");
		testY.loadARFF("data/test_lab.arff");
		
		// Normalize training and test data
		trainX.scale(0.00390625); // divide by 256
		testX.scale(0.00390625);
		
		//System.out.println("Building neural net . . . ");
		/// Build Neural Net

		NeuralNet DigitPredict = new NeuralNet(1.0, 1);
		LayerLinear one = new LayerLinear(784, 80);
		DigitPredict.layerCollection.add(one);
		LayerTanh two = new LayerTanh(80);
		DigitPredict.layerCollection.add(two);
		LayerLinear three = new LayerLinear(80, 30);
		DigitPredict.layerCollection.add(three);
		LayerTanh four = new LayerTanh(30);
		DigitPredict.layerCollection.add(four);
		LayerLinear five = new LayerLinear(30, 10);
		DigitPredict.layerCollection.add(five);
		LayerTanh six = new LayerTanh(10);
		
		// Set up gradient and blame Vecs
		for(Layer layer : DigitPredict.layerCollection) {
			layer.gradients.fill(0.0);
			layer.blame.fill(0.0);
			DigitPredict.layerGradients.add(layer.gradients);
		}
		
		
		//System.out.println("Initializing weights . . . ");
		DigitPredict.initWeights();
		//System.out.println("First set of weights: " + DigitPredict.layerWeights.get(0).get(0));
		
		/// Train model iteratively until reach a low misclassification rate
		int startMisclass = DigitPredict.countMisclassifications(testX, testY, true);
		long startTime = System.nanoTime();
		System.out.println(startMisclass + ", " + 0.0);
		
		
		for(int i = 0; i < 1000; i++) {
			//System.out.println("Epoch " + (i+1));
			
			//System.out.println("     Training neural net . . . ");
			DigitPredict.train(trainX, onehotTrain);
			
			
			//System.out.println("     Testing neural net . . . ");
			int misclass = DigitPredict.countMisclassifications(testX, testY, true);
			System.out.print(misclass);
			
			long currentTime = System.nanoTime();
			System.out.println(", " + (currentTime - startTime)/1000000000.0);

		}	
	
	/*
	System.out.println("9050");
	System.out.println("963");
	System.out.println("761");
	System.out.println("634");
	System.out.println("558");
	System.out.println("495");
	System.out.println("485");
	System.out.println("433");
	System.out.println("416");
	System.out.println("403");
	System.out.println("372");
	System.out.println("361");
	System.out.println("342");
	System.out.println("317");
	System.out.println("325");
	System.out.println("309");
	System.out.println("302");
	System.out.println("306");
	System.out.println("294");
	System.out.println("296");
	System.out.println("283");
	*/
	}
}
