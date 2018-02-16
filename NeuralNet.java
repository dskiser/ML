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
		//System.out.println("Targets: " + targets);
		//System.out.println("Activation: " + predicted);
		layerCollection.get(indexOutLayer--).blame = targets.sum(-1, predicted);
		//System.out.println("BackProp Blame: " + layerCollection.get(indexOutLayer+1).blame);
		//System.out.println(); 
		
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
			layerGradients.add(i, layerCollection.get(i).updateGradient(input, current_weights));
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
			gradient[i] = (-0.5) * ((SSEstepUp - SSEstepDown) / stepSize); 
						// multiply be -0.5 to make equivalent to derivative method
		}                        
		Vec gradients = new Vec(gradient);
		this.layerGradients.add(gradients);	
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
		
		test.updateGradient(x);
		Vec gradient = test.layerGradients.get(0);	

		// Compare gradients
		if(gradientfd.squaredDistance(gradient) >= Math.pow(10, -6))
			throw new TestFailedException("testGradient");
	}
	 
	public String name() {
		return("My Neural Net");
	}		
}
	
	
