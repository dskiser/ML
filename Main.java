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
			LayerLinear.testActivate(false);
			NeuralNet.testGradient();
			//NeuralNet.testCrossValidate();
			//NeuralNet.testCrossValidate2();
			NeuralNet.testComputeSumSquaredError();
			//NeuralNet.testRefineWeights();
			
		} catch (TestFailedException exc) {
			System.out.println(exc);
		}
		//testLearner(new BaselineLearner());
		//testLearner(new RandomForest(50));
		//Matrix.testShuffle();
		
		/// Predict hypothyroidism
		
		/// Load data
		Matrix data = new Matrix();
		data.loadARFF("data/hypothyroid.arff");
		//System.out.println(data);
		data.shuffle(); // shuffle in case there is any order in original data set
		
		/// Divide data into features and labels and into training set and testing set
		int test_size = data.rows()/10;
		int train_size = data.rows() - test_size;
		//System.out.println("test data set size: " + test_size);
		Matrix testfeat = new Matrix(test_size, data.cols() - 1);
		Matrix testlab = new Matrix(test_size, 1);
		Matrix trainfeat = new Matrix(data.rows() - test_size, data.cols() - 1);
		Matrix trainlab = new Matrix(data.rows() - test_size, 1);
		
		
		testfeat.copyBlock(0, 0, data, 0, 0, test_size, data.cols()-1);
		testlab.copyBlock(0, 0, data, 0, data.cols() - 1, test_size, 1);
		trainfeat.copyBlock(0, 0, data, test_size , 0, data.rows() - test_size, data.cols() -1);
		trainlab.copyBlock(0, 0, data, test_size, data.cols()-1, data.rows() - test_size, 1);
		
		/// Baseline Learner
		BaselineLearner base = new BaselineLearner();
		base.train(trainfeat, trainlab);
		double train_base_misclass = (double) base.countMisclassifications(trainfeat, trainlab) / (double) train_size;
		double base_misclass = (double) base.countMisclassifications(testfeat, testlab) / (double) test_size;
		//System.out.println("Baseline Learner: " + train_base_misclass + ", " + base_misclass);
		
		/// Build Neural Net
		NeuralNet NN = new NeuralNet(4.0, 1);
		LayerLinear one = new LayerLinear(33, 800);
		NN.layerCollection.add(one);
		LayerTanh two = new LayerTanh(800);
		NN.layerCollection.add(two);
		LayerLinear three = new LayerLinear(800, 600);
		NN.layerCollection.add(three);
		LayerTanh three_third = new LayerTanh(600);
		NN.layerCollection.add(three_third);
		LayerLinear three_half = new LayerLinear(600, 300);
		NN.layerCollection.add(three_half);
		LayerTanh four = new LayerTanh(300);
		NN.layerCollection.add(four);
		LayerLinear seven = new LayerLinear(300, 200);
		NN.layerCollection.add(seven);
		LayerTanh eight = new LayerTanh(200);
		NN.layerCollection.add(eight);
		LayerLinear nine = new LayerLinear(200, 100);
		NN.layerCollection.add(nine);
		LayerTanh ten = new LayerTanh(100);
		NN.layerCollection.add(ten);
		LayerLinear five = new LayerLinear(100, 4);
		NN.layerCollection.add(five);
		LayerTanh six = new LayerTanh(4);
		NN.layerCollection.add(six);
		// Set up gradient and blame Vecs
		for(Layer layer : NN.layerCollection) {
			layer.gradients.fill(0.0);
			layer.blame.fill(0.0);
			NN.layerGradients.add(layer.gradients);
		}
		// Initialize weights
		NN.initWeights();
		
		/// Set up filter
		Filter hypo = new Filter(NN, true, true, true);
		
		/// Train model iteratively until reach a low misclassification rate

		hypo.train(trainfeat, trainlab);
		hypo.initialize = false;
		DecimalFormat df = new DecimalFormat("####.######");
		double start_train_misclass = ((double) hypo.countMisclassifications(trainfeat, trainlab) / (double) train_size);
		double startMisclass = (double) hypo.countMisclassifications(testfeat, testlab)/ (double) test_size;
		System.out.println(df.format(start_train_misclass) + ", " /*+ df.format(startMisclass) + ", "*/ + 0.0);
		
		startTime = System.nanoTime();
		double window_size = 5;
		double k = 0.02;
		boolean convergence = false;
		int extra = 10;
		int count = 0;
		double convergeTime = 0;
		for(int i = 0; i < 400; i++) {
			if(convergence) {
				count++;
				if(count >= extra) break;
			}
			//System.out.println("Pattern " + (i+1));
			hypo.train(trainfeat, trainlab);
			
			double train_misclass = (double) hypo.countMisclassifications(trainfeat, trainlab)/ (double) train_size;
			System.out.print(df.format(train_misclass));
			double test_misclass = (double) hypo.countMisclassifications(testfeat, testlab) / (double) test_size;
			//System.out.print(", " + df.format(test_misclass));
			long currentTime = System.nanoTime();
			System.out.println(", " + /*(i+1));*/df.format((currentTime - startTime)/1000000000.0));
			if((i >= window_size) & ((i % window_size) == 0)) {
				if((startMisclass - test_misclass <= k) & !convergence) {
					convergeTime = /*(double) (i+1);*/(double) (currentTime - startTime)/1000000000.0;
					convergence = true;
				}
				startMisclass = test_misclass;
			}
		}
		System.out.println("Converge time: " + convergeTime);
	}
}
