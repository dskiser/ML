// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.lang.Math;
import java.util.ArrayList;

abstract class SupervisedLearner 
{
	protected ArrayList<LayerLinear> layerCollection;
	protected ArrayList<Vec> layerWeights;
	
	SupervisedLearner() {
		layerCollection = new ArrayList<LayerLinear>(5);
		layerWeights = new ArrayList<Vec>(5);
	}
	
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels);

	/// Make a prediction
	abstract Vec predict(Vec in);

	/// Measures the misclassifications with the provided test data
	int countMisclassifications(Matrix features, Matrix labels)
	{
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		int mis = 0;
		for(int i = 0; i < features.rows(); i++)
		{
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = labels.row(i);
			for(int j = 0; j < lab.size(); j++)
			{
				if(pred.get(j) != lab.get(j))
					mis++;
			}
		}
		return mis;
	}
	
	/// Compute sum squared error
	double computeSumSquaredError(Matrix training, Matrix predicted) {
		if(training.rows() != predicted.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		double sumSquaredError = 0.0;
		for(int i=0; i < training.rows(); i++)
		{
			Vec tr = training.row(i);
			Vec pred = predicted.row(i);
			for(int j = 0; j < tr.size(); j++)
			{
				double errorSquared = Math.pow((tr.get(j) - pred.get(j)), 2);
				sumSquaredError += errorSquared;
			}
		}
		return sumSquaredError;
	}
		
	
	/// Cross validation with m repetitions and n folds
	double crossValidate(Matrix X, Matrix Y, int reps, int folds) {
		if(X.rows() != Y.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		
		// Join Matrix X and Y for shuffling
		Matrix data = new Matrix(X.rows(), X.cols() + Y.cols());
		data.copyBlock(0, 0, Y, 0, 0, Y.rows(), Y.cols());
		data.copyBlock(0, Y.cols(), X, 0, 0, X.rows(), X.cols());
		
		// Loop through repetitions
		double totalSSE = 0.0;
		for(int i=0; i<reps; i++) {
			// Shuffle data for each rep
			data.shuffle();
			// Split Matrix X and Y
			Matrix Xshuf = new Matrix(X.rows(), X.cols());
			Matrix Yshuf = new Matrix(Y.rows(), Y.cols());
			Xshuf.copyBlock(0, 0, data, 0, Y.cols(), X.rows(), X.cols());
			Yshuf.copyBlock(0, 0, data, 0, 0, Y.rows(), Y.cols());
			
			// Create Vec of row numbers to keep track of observations
			// in training set
			double[] row_nums = new double[X.rows()];
			for(int j=0; j<X.rows(); j++) row_nums[j] = j;
			Vec rows = new Vec(row_nums);
			int testSetSize = X.rows() / folds;
			int remainder = X.rows() % folds;
			
			// Loop through folds
			for(int j=0; j<folds; j++) {
				// Create test Matrices
				int testStartRow = j * testSetSize;
				if(j == folds - 1) testSetSize += remainder; // on last fold, add remaining rows to test set
				
				// Training Matrices before test Matrices are removed
				Matrix trainX = new Matrix();
				Matrix trainY = new Matrix();
				trainX.copy(Xshuf);
				trainY.copy(Yshuf);
				
				Matrix testX = new Matrix(0, X.cols());
				Matrix testY = new Matrix(0, Y.cols());
				for(int k=0; k<testSetSize; k++) {
					// add rows to test X
					double[] xrow = trainX.removeRow(testStartRow);
					testX.takeRow(xrow);
					// add rows to test Y
					double[] yrow = trainY.removeRow(testStartRow);
					testY.takeRow(yrow);
				}

					
				// Train model
				this.train(trainX, trainY);
				//System.out.println(this.layerWeights);
				
				// Use model to predict Y on test Matrices
				Matrix Y_hat = new Matrix(0, Y.cols());
				for(int k=0; k<testY.rows(); k++) {
					Vec Xrow = testX.row(k);
					Vec Yrow = this.predict(Xrow);
					Y_hat.takeVec(Yrow);
				}
				
				/* To compare actual response versus predicted response
				for(int p=0; p<testY.rows(); p++) {
					System.out.println(testY.get(p,0) + " " + Y_hat.get(p,0));
				}
				*/
				// Compute sumSquareError of each fold
				totalSSE += computeSumSquaredError(testY, Y_hat);
						
			}
		}
		double avgSSE = totalSSE / reps;
		double MSE = avgSSE / data.rows();
		double RMSE = Math.sqrt(MSE);
		return RMSE;
	}
	
	/// Test crossValidate()
	static void testCrossValidate()
			throws TestFailedException {
		NeuralNet test = new NeuralNet();
		LayerLinear testlinear = new LayerLinear(3, 2);
		test.layerCollection.add(testlinear);
		
		// Test data
		Matrix X = new Matrix(0, 3);
		double[] x = { 1, 2, 3 };
		X.takeRow(x);
		double[] y = { 0, 0, 0 };
		X.takeRow(y);		
		double[] z = { 5, 3, 1 };
		X.takeRow(z);
		double[] w = { 4, 2, 9 };
		X.takeRow(w);
		double[] u = { 1, 4, 7 };
		X.takeRow(u);
		double[] v = { 2, 5, 8 };
		X.takeRow(v);
		double[] c = { 9, 7, 2 };
		X.takeRow(c);
		double[] d = { 8, 1, 5 };
		X.takeRow(d);
		double[] g = { 7, 7, 7 };
		X.takeRow(g);
		
		Matrix Y = new Matrix(0, 2);
		double[] p = { 3, 7 };
		Y.takeRow(p);	
		double[] q = { 1, 8 };
		Y.takeRow(q);
		double[] r = { 5, 3 };
		Y.takeRow(r);
		double[] s = { 2, 1 };
		Y.takeRow(s);
		double[] t = { 6, 4 };
		Y.takeRow(t);
		double[] a = { 9, 7 };
		Y.takeRow(a);	
		double[] b = { 0, 0 };
		Y.takeRow(b);	
		double[] e = { 5, 5 };
		Y.takeRow(e);
		double[] f = { 4, 4 };
		Y.takeRow(f);		
		
		double RMSE = test.crossValidate(X, Y, 2, 2);
		if(RMSE < 1 || RMSE > 75)
			throw new TestFailedException("testCrossValidate");
	}

	static void testComputeSumSquaredError() 
			throws TestFailedException {
		NeuralNet test = new NeuralNet();
		LayerLinear testlinear = new LayerLinear(3, 2);
		test.layerCollection.add(testlinear);
		
		Matrix pred = new Matrix(2, 2);
		pred.fill(1.0);
		Matrix actual = new Matrix(2, 2);
		actual.fill(2.0);
		
		double SSE = test.computeSumSquaredError(actual, pred);
		if(SSE != 4.0) throw new TestFailedException("computeSumSquaredError");
	}
	
	static void testCrossValidate2() 
			throws TestFailedException {
		Matrix features = new Matrix(3,1);
		features.fill(0.0);
		Matrix labels = new Matrix(0, 1);
		for(int i=2; i<7; i+=2) {
			double[] value = { (double)i };
			labels.takeRow(value);
		}
				
		NeuralNet test = new NeuralNet();
		LayerLinear testlinear = new LayerLinear(1, 1);
		test.layerCollection.add(testlinear);
		
		double RMSE = test.crossValidate(features, labels, 2, 3);
		if(RMSE != Math.sqrt(6))
			throw new TestFailedException("testCrossValidate2");
	}
}
