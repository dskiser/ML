import java.util.ArrayList;

class NeuralNet extends SupervisedLearner {

	NeuralNet() {
		super();
	}
	
	public Vec predict(Vec x) {
		Vec prev_activation = x;
		for(int i = 0; i < layerCollection.size(); i++) {
			layerCollection.get(i).activate(layerWeights.get(i), x);
			prev_activation = layerCollection.get(i).activation;
		}
		Vec final_activation = prev_activation;
		return final_activation;
	}
	
	public void train(Matrix X, Matrix Y) {
		Vec weights = ((LayerLinear)layerCollection.get(0)).ordinary_least_squares(X, Y);
		layerWeights.add(0, weights);
	}
	
	// Compute blame for each layer in Neural Network
	public void backProp(Vec weights, Vec targets) {
		int indexOutLayer = layerCollection.size()-1;
		Vec predicted = layerCollection.get(indexOutLayer).activation;
		
		// Compute blame for output layer
		layerCollection.get(indexOutLayer--).blame = targets.sum(-1, predicted);
		//System.out.println("Blame: " + layerCollection.get(indexOutLayer+1).blame);
		// Compute blame for each subsequent layer
		for(int i = indexOutLayer; i >= 0; i--) {
			Vec current_weights = layerWeights.get(i);
			Vec prevBlame = layerCollection.get(i-1).blame;
			layerCollection.get(i).backprop(current_weights, prevBlame); 
		}		
	}
	
	// Use activation of previous layer and blame of current layer to compute
	// gradient vector of each layer 
	public void updateGradient(Vec x) {
		// Update weights of first layer using original data
		layerGradients.add(0, layerCollection.get(0).updateGradient(x, layerWeights.get(0)));
		
		// Update subsequent layer weights using activation from previous layers
		for(int i = 1; i < layerCollection.size(); i++) {
			// Input is activation from previous layer
			Vec input = layerCollection.get(i-1).activation;
			Vec current_weights = layerWeights.get(i);
			Vec gradients = layerCollection.get(i).updateGradient(input, current_weights);
			layerGradients.add(i, gradients);
		}
	}
	
	public void updateGradientFD(Vec x, Vec targets) {
		LayerLinear layer = (LayerLinear) this.layerCollection.get(0);
		Vec weights = layerWeights.get(0);
			
		// Determine initial SSE
		Vec error = layer.blame;
		double SSE = 0.0;
		for(int i = 0; i < error.size(); i++) { 
			double squaredError = Math.pow(error.get(i), 2);
			SSE += squaredError;
		}
		
		// Loop through each weight and update gradient
		double[] gradient = new double[weights.size()];
		double stepSize = 1 * Math.pow(10, -6);
		for(int i = 0; i < weights.size(); i++) {
			
			// Step up
			weights.set(i, weights.get(i) + (0.5*stepSize));
			this.predict(x);
			this.backProp(weights, targets);
			// Determine SSE of step up
			Vec error1 = layer.blame;
			double SSEstepUp = 0.0;
			for(int j = 0; j < error1.size(); j++) { 
				double squaredError1 = Math.pow(error1.get(j), 2);
				SSEstepUp += squaredError1;
			}
			
			// Step down
			weights.set(i, weights.get(i) + (-1*stepSize));
			layer.activate(weights, x);
			this.backProp(weights, targets);
			// Determine SSE of step down
			Vec error2 = layer.blame;
			double SSEstepDown = 0.0;
			for(int j = 0; j < error2.size(); j++) { 
				double squaredError2 = Math.pow(error2.get(j), 2);
				SSEstepDown += squaredError2;
			}
			weights.set(i, weights.get(i) + (0.5*stepSize));
			gradient[i] = -0.5 * ((SSEstepUp - SSEstepDown) / stepSize); 
						// multiply be -0.5 to make equivalent to derivative method
		}                        
		Vec gradients = new Vec(gradient);
		this.layerGradients.add(gradients);	
	}
	
	public void initWeights() {
		int numLayers = layerCollection.size();
		for(int i = 0; i < numLayers; i++) {
			Layer currentLayer = layerCollection.get(i);
			int numWeights = (currentLayer.num_inputs + 1) * currentLayer.num_outputs;
			Vec weights = new Vec(numWeights);
			for(int j = 0; j < weights.size(); j++) {
				double randWeight = Math.max(0.03, 1/currentLayer.num_inputs) * MyRandom.getgaussian();
				weights.set(j, randWeight);
			}
			layerWeights.add(i, weights);
		}
	}
	
	public void refineWeights(Vec x, Vec y, double learning_rate) {
		Vec weights = layerWeights.get(0);;
		this.predict(x);
		this.backProp(weights, y); // compute blame for each layer
		this.updateGradient(x); // compute gradients for each layer
		
		// Update weights in each layer
		int numLayers = layerCollection.size();
		for(int i = 0; i < numLayers; i++) {
			// Loop through each weight vector (and corresponding gradient vector)
			Vec current_weights = layerWeights.get(i);
			for(int j = 0; j < weights.size(); j++) {
				double new_weight = current_weights.get(j) + (layerGradients.get(i).get(j) * learning_rate);
				current_weights.set(j, new_weight);
			}
			layerWeights.set(i, current_weights);
		}
	}
	
	public static void testRefineWeights() 
		throws TestFailedException {
		// Generate random data set
		int inputs = 2;
		int outputs = 2;
		int patterns = 5;
		
		Matrix X = new Matrix(0, inputs);
		for(int n = 0; n < patterns; n++) {
			double[] x_row = new double[inputs];
			for(int p = 0; p < (inputs); p++) {
				x_row[p] = MyRandom.getdouble();
			}
			X.takeRow(x_row);
		}
		
		Matrix Y = new Matrix(0, outputs);
		for(int n = 0; n < patterns; n++) {
			double[] y_row = new double[outputs];
			for(int p = 0; p < outputs; p++) {
				y_row[p] = MyRandom.getdouble();
			}
			Y.takeRow(y_row);
		}
		
		// Generate weights with OLS			
		LayerLinear OLSlayer = new LayerLinear(inputs, outputs);
		Vec OLSweights = OLSlayer.ordinary_least_squares(X, Y);
		//System.out.println("OLS weights:");
		//System.out.println(OLSweights);
		
		// Generate weights with gradient descent
		NeuralNet GDtest = new NeuralNet();
		LayerLinear GDlayer = new LayerLinear(inputs, outputs);
		GDtest.layerCollection.add(GDlayer);
		GDtest.initWeights();
		//System.out.println("Initial weights: ");
		//System.out.println(GDtest.layerWeights.get(0));
		for(int i = 0; i < 100000; i++) {
			int row_index = MyRandom.getinteger(patterns);
			Vec x_sample = X.row(row_index);
			Vec y_sample = Y.row(row_index);
			GDtest.refineWeights(x_sample, y_sample, 0.001);
		}
		Vec GDweights = GDtest.layerWeights.get(0);
		//System.out.println("GD weights: ");
		//System.out.println(GDweights);
		if(GDweights.squaredDistance(OLSweights) >= 0.5)
			throw new TestFailedException("testRefineWeights");
	}
		
	public static void testGradient() 
		throws TestFailedException {
			
		int outputs = 2;
		int inputs = 3;
		NeuralNet test = new NeuralNet();	
		LayerLinear layer = new LayerLinear(inputs, outputs);
		test.layerCollection.add(layer);
			
		// Generate random weights
		double[] weight_values = new double[(inputs + 1) * outputs];
		for(int i = 0; i < weight_values.length; i++) {
			weight_values[i] = -2 + (4 * MyRandom.getdouble());
		}
		Vec weights = new Vec(weight_values);
		test.layerWeights.add(weights);
		
		// Generate random input
		double[] x_values = new double[inputs];
		for(int i = 0; i < x_values.length; i++) {
			x_values[i] = -2 + (4 * MyRandom.getdouble());
		}
		Vec x = new Vec(x_values);
		
		// Generate random targets
		double[] y_values = new double[outputs];
		for(int i = 0; i < y_values.length; i++) {
			y_values[i] = -2 + (4 * MyRandom.getdouble());
		}
		Vec y = new Vec(y_values);	
		
		// Call predict() to generate activation
		test.predict(x);
		// Call backProp() to generate blame
		test.backProp(weights, y);
		
		// Find gradient using both methods
		test.updateGradientFD(x, y);
		Vec gradientfd = test.layerGradients.get(0);
		//System.out.println("gradientfd");
		//System.out.println(gradientfd);
		
		test.updateGradient(x);
		Vec gradient = test.layerGradients.get(0);	
		//System.out.println("gradient");
		//System.out.println(gradient);
		// Compare gradients
		if(gradientfd.squaredDistance(gradient) >= Math.pow(10, -6))
			throw new TestFailedException("testGradient");
	}
	 
	public String name() {
		return("My Neural Net");
	}		
}
	
	
