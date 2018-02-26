class TestFailedException extends Exception {
	String test;
	
	TestFailedException(String test_name) {
		test = test_name;
	}
	
	public String toString() {
		return test + " failed";
	}
}

class LayerLinear extends Layer {
	
	LayerLinear(int inputs, int outputs) {
		super(inputs, outputs);
	}
	
	public void activate(Vec weights, Vec x) {
		int x_length = num_inputs;
		int y_length = num_outputs;
		// Define and store values in b
		Vec b = new Vec(weights, 0, y_length);
		// Define Mx
		Vec Mx = new Vec(y_length);
		// For each value in x, calculate M_row_i*x_i
		int weight_count = y_length;
		for(int i=0; i<y_length; i++) {
			Vec M_row = new Vec(weights, weight_count, x_length);
			weight_count += x_length;
			double Mx_entry = M_row.dotProduct(x);
			Mx.set(i, Mx_entry);
		}
		// Mx + b
		Mx.add(b);
		activation = new Vec(Mx, 0, y_length);
	}
	
	public Vec backprop(Vec weights) {
		// Remove b from weights
		Vec mweights = new Vec(weights, this.num_outputs, weights.size() - this.num_outputs);
		
		// Convert Vec mweights into transposed matrix weights
		if(mweights.size() % this.num_outputs != 0)
			throw new IllegalArgumentException("Incorrect number of weights");
		int numColumns = mweights.size() / this.num_outputs;
		Matrix M = new Matrix(mweights, this.num_outputs, numColumns);
		
		
		// Convert Vec lame into Matrix
		Matrix BlameMatrix = new Matrix(this.blame, this.blame.size(), 1);
		
		// Calculate blame of current layer
		Matrix PrevBlame = Matrix.multiply(M, BlameMatrix, true, false);
		Vec prevBlame = new Vec(0, PrevBlame);
		
		return prevBlame;
	}
	
	public void updateGradient(Vec x, Vec weights) {
		// Create bias Vec
		int interceptLength = num_outputs;
		Vec b = new Vec(weights, 0, interceptLength);
		
		// Create weight Vec
		Vec m = new Vec(weights, interceptLength, weights.size()-interceptLength);
		// Find gradients for bias
		int k = 0;
		for(int i = 0; i < this.blame.size(); i++) {
				gradients.set(k, gradients.get(k) + this.blame.get(i));
				k++;
		}
			
		// Find gradients for weights
		for(int j = 0; j < this.blame.size(); j++) {
			for(int i = 0; i < x.size(); i++) {
				gradients.set(k, gradients.get(k) + x.get(i)*this.blame.get(j));
				k++;
			}
		}
	}		
		
	public Vec ordinary_least_squares(Matrix X_start, Matrix Y) {

		Matrix X = new Matrix(X_start.rows(), X_start.cols() + 1);
		
		// Add intercept column to X Matrix
		Matrix Inter = new Matrix(X.rows(), 1);
		Inter.fill(1);
		X.copyBlock(0, 0, Inter, 0, 0, X.rows(), Inter.cols());
		
		// Add X_start Matrix to X Matrix
		X.copyBlock(0, 1, X_start, 0, 0, X.rows(), X_start.cols());
		
		// Compute beta_hat
		Matrix xTx = Matrix.multiply(X, X, true, false);
		Matrix inverseXtX = xTx.pseudoInverse();
		Matrix XtY = Matrix.multiply(X, Y, true, false);
		Matrix beta_hat = Matrix.multiply(inverseXtX, XtY, false, false);
		
		// Fill Vec of weights
		Vec weights = new Vec(Y.cols(), beta_hat);
		return weights;
	}
	
	public static void testActivate(boolean verbose)
		throws TestFailedException {
		// Test values
		double[] weight_values = { 1, 3, 2, 3, 0, 1, 0, 2 };
		double[] x_values = { 1, 0, 2 };
		double[] answers = { 3, 8 };
		Vec correct_answer = new Vec(answers);
		
		// Create vectors
		Vec x = new Vec(x_values);
		Vec weights = new Vec(weight_values);
		
		// Test activate()
		LayerLinear testLayer = new LayerLinear(3, 2);
		testLayer.activate(weights, x);
		if(testLayer.activation.size() != correct_answer.size())
			throw new TestFailedException("testActivate");
		for(int i=0; i<correct_answer.size(); i++)
			if(testLayer.activation.get(i) != correct_answer.get(i))
				throw new TestFailedException("testActivate");
		
		if(verbose) {
			System.out.println("activation: " + testLayer.activation);
			System.out.println("correct answer: " + correct_answer);
		}
	}
	
	public static void testOrdinary_least_squares(
							int inputs, int outputs, boolean verbose) 
		throws TestFailedException {
		
		int weight_num = outputs + (inputs * outputs);
		// Generate number of random weights specified by weight_num
		Vec weights = new Vec(weight_num);
		for(int i=0; i<weight_num; i++) {
			double weight = -10 + (20 * MyRandom.getdouble());
			weights.set(i, weight);
		}
		
		// Generate random feature matrix X (50 x inputs)
		Matrix X = new Matrix(0, inputs);
		for(int i=0; i<50; i++) {
			double[] row = new double[inputs];
			for(int j=0; j<inputs; j++) row[j] = -25 + (50 * MyRandom.getdouble());
			X.takeRow(row);
		}
		
		// Create label Matrix Y from weights and X
		LayerLinear testOLS = new LayerLinear(inputs, outputs); 
		Matrix Y = new Matrix(0, outputs);
		Matrix Xcopy = new Matrix();
		Xcopy.copy(X);
		int start_rows = Xcopy.rows();
		for(int i=0; i<start_rows; i++) {
			Vec X_row = new Vec(Xcopy.removeRow(0));
			testOLS.activate(weights, X_row);
			double[] Y_row = new double[Y.cols()];
			for(int j=0; j<Y.cols(); j++) {
				Y_row[j] = testOLS.activation.get(j);
			}
			Y.takeRow(Y_row);
		}
				
		// Add random noise to Y
		for(int i=0; i<Y.m_data.size(); i++) {
			double[] column = Y.m_data.get(i);
			double[] noisy_column = new double[column.length];
			for(int j=0; j<column.length; j++) {
				double randomIncrement = -25 + (50 * MyRandom.getgaussian());
				noisy_column[j] = column[j] + randomIncrement;
			}
			Y.m_data.set(i, noisy_column);
		}	
		
		// Generate new weights with ordinary_least_squares()
		// Vec new_weights;
		Vec new_weights = testOLS.ordinary_least_squares(X, Y);
		
		// Visually compare new weights with original weights
		if(verbose) {
			System.out.println("Original weights:");
			System.out.println(weights);
			System.out.println("New weights:");
			System.out.println(new_weights);
		}
		
		// Compare new weights with original weights
		if(weights.squaredDistance(new_weights) > 3000)
			throw new TestFailedException("testOrdinary_least_squares");
	}
}
