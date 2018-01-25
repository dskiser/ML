// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.lang.Math;

abstract class SupervisedLearner 
{
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
		for(int i=0; i<reps; i++) {
			// Shuffle data for each rep
			data.shuffle();
			// Split Matrix X and Y
			Matrix Xshuf = new Matrix(X.rows(), X.cols());
			Matrix Yshuf = new Matrix(X.rows(), Y.cols());
			Xshuf.copyBlock(0, 0, data, 0, Y.cols(), X.rows(), X.cols());
			Yshuf.copyBlock(0, 0, data, 0, 0, Y.rows(), Y.cols());
			for(int j=0; j<folds; j++) {}
		}
		
		
		return 0.0;
	}
	
	/// Test crossValidate()
	static void testCrossValidate() {
		NeuralNet test = new NeuralNet();
		
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
		
		test.crossValidate(X, Y, 3, 0);
	}
				
}
