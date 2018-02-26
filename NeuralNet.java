import java.util.ArrayList;

class NeuralNet extends SupervisedLearner {
	
	protected double momentum;
	protected int batch_size;
	protected int count; // counts the patterns NN is trained on, in order to implement batch_size

	NeuralNet(double effMiniBatch, int miniBatch) {
		super();
	
		momentum = 1 - 1 / effMiniBatch;
		batch_size = miniBatch;
		count = 0;	
	}
	
	public Vec predict(Vec x) {
		Vec prev_activation = x;
		for(int i = 0; i < layerCollection.size(); i++) {
			layerCollection.get(i).activate(layerWeights.get(i), prev_activation);
			prev_activation = layerCollection.get(i).activation;
		}
		Vec final_activation = prev_activation;
		return final_activation;
	}
	
	public void train(Matrix X, Matrix Y) {
		int cycles = 1;
		for(int j = 0; j < cycles; j++) {
			// Create array of shuffled indexes
			int patterns = X.rows();
			int[] randindex = new int[patterns];
			for(int i=0; i<randindex.length; i++) {
				randindex[i] = i;
			}
			for(int i=randindex.length-1; i>0; i--) {
				int temp = randindex[i];
				int swap_index = MyRandom.getinteger(i);
				randindex[i] = randindex[swap_index]; 
				randindex[swap_index] = temp;
			}
		
			// Randomly visit each row of X and Y, refining weights for each visit
			for(int i : randindex) {
				Vec x = X.row(i);
				Vec y = Y.row(i);
				this.refineWeights(x, y, 0.025);
			}
		}
	}
	
	public void refineWeights(Vec x, Vec y, double learning_rate) {
		//System.out.println("In refineWeights: " + layerWeights.get(0).get(0));
		Vec weights = layerWeights.get(0);
		this.predict(x);
		this.backProp(weights, y); // compute blame for each layer
		/*
		System.out.println("Before update gradients: ");
		for(int i = 0; i < 5; i++) {
			System.out.print(layerGradients.get(0).get(i));
		}
		System.out.println();
		*/
		// Add momentum
		for(Vec grad : layerGradients) {
			if(momentum == 0.0) grad.fill(0.0);
			else grad.scale(momentum);
		}
		this.updateGradient(x); // compute gradients for each layer
		/*
		System.out.println("After update gradients: ");
		for(int i = 0; i < 5; i++) {
			System.out.print(layerGradients.get(0).get(i));
		}
		System.out.println();
		*/
		// Update weights in each layer
		if(count++ == batch_size) {
			count = 0;
			int numLayers = layerCollection.size();
			for(int i = 0; i < numLayers; i++) {
				// Loop through each weight vector (and corresponding gradient vector)
				Vec current_weights = layerWeights.get(i);
				for(int j = 0; j < current_weights.size(); j++) {
					double new_weight = current_weights.get(j) + (layerGradients.get(i).get(j) * learning_rate);
					current_weights.set(j, new_weight);
				}
				layerWeights.set(i, current_weights);
			}
		}
		/*
		System.out.println("After update weights: ");
		for(int j = 0; j < 5; j++) {
			System.out.print(layerGradients.get(0).get(j));
		}
		System.out.println();
		*/
	}
	
	// Compute blame for each layer in Neural Network
	public void backProp(Vec weights, Vec targets) {
		int indexOutLayer = this.layerCollection.size()-1;
		Vec predicted = this.layerCollection.get(indexOutLayer).activation;
		
		// Compute blame for output layer
		layerCollection.get(indexOutLayer).blame = targets.sum(-1, predicted);
		//System.out.println("Blame: " + layerCollection.get(indexOutLayer+1).blame);
		// Compute blame for each subsequent layer
		for(int i = indexOutLayer; i >= 1; i--) {
			//Vec current_weights = layerWeights.get(i);
			//Vec prevBlame = layerCollection.get(i).backprop(layerWeights.get(i));
			layerCollection.get(i-1).blame = layerCollection.get(i).backprop(layerWeights.get(i)); 
		}		
	}
	
	// Use activation of previous layer and blame of current layer to compute
	// gradient vector of each layer 
	public void updateGradient(Vec x) {
					
		// Update weights of first layer using original data
		layerCollection.get(0).updateGradient(x, layerWeights.get(0));
		layerGradients.set(0, layerCollection.get(0).gradients);
		
		// Update subsequent layer weights using activation from previous layers
		for(int i = 1; i < layerCollection.size(); i++) {
			// Input is activation from previous layer
			Vec input = layerCollection.get(i-1).activation;
			Vec current_weights = layerWeights.get(i);
			layerCollection.get(i).updateGradient(input, current_weights);
			Vec netgradients = layerCollection.get(i).gradients;
			layerGradients.set(i, netgradients);
		}
	}
	
	public void updateGradientFD(Vec x, Vec targets) {
		//System.out.println("x: " + x);
		int numLayers = this.layerCollection.size();
		for(int h = 0; h < numLayers; h++) {
			Vec prevActivation;
			if(h == 0)
				prevActivation = x;
			else prevActivation = this.layerCollection.get(h-1).activation;
			//System.out.println("prevActivation: " + prevActivation);
			Layer layer = this.layerCollection.get(h);
			Vec weights = layerWeights.get(h);
			//System.out.println("Weights: " + weights);
			
			// Determine target for each layer
			Vec error = layer.blame;
			//System.out.println("error: " + error);
			Vec layerTarget = layer.activation.sum(1, layer.blame);
		
			// Loop through each weight and update gradient
			double[] gradient = new double[weights.size()];
			double stepSize = 1 * Math.pow(10, -6);
			for(int i = 0; i < weights.size(); i++) {
				//System.out.println("Find gradient for weight " + i);
				// Step up
				weights.set(i, weights.get(i) + (0.5*stepSize));
				layer.activate(weights, prevActivation);
				//System.out.println("New activation: " + layer.activation);
				// Determine SSE of step up
				Vec error1 = layerTarget.sum(-1, layer.activation);
				double SSEstepUp = 0.0;
				for(int j = 0; j < error1.size(); j++) { 
					double squaredError1 = Math.pow(error1.get(j), 2);
					SSEstepUp += squaredError1;
				}
				//System.out.println("SSEstepUp: " + SSEstepUp);
				// Step down
				weights.set(i, weights.get(i) + (-1*stepSize));
				layer.activate(weights, prevActivation);
				// Determine SSE of step down
				Vec error2 = layerTarget.sum(-1, layer.activation);
				double SSEstepDown = 0.0;
				for(int j = 0; j < error2.size(); j++) { 
					double squaredError2 = Math.pow(error2.get(j), 2);
					SSEstepDown += squaredError2;
				}
				weights.set(i, weights.get(i) + (0.5*stepSize));
				gradient[i] = -0.5 * ((SSEstepUp - SSEstepDown) / stepSize); 
						// multiply by -0.5 to make equivalent to derivative method
				//System.out.println("Gradient: " + gradient[i]);
			}                        
			Vec gradients = new Vec(gradient);
			this.layerGradients.add(gradients);
		}	
	}
	
	public void initWeights() {
		int numLayers = layerCollection.size();
		for(int i = 0; i < numLayers; i++) {
			Layer currentLayer = layerCollection.get(i);
			Vec weights;
			if(currentLayer instanceof LayerLinear) {
				int numWeights = (currentLayer.num_inputs + 1) * currentLayer.num_outputs;
				weights = new Vec(numWeights);
				for(int j = 0; j < weights.size(); j++) {
					double randWeight = Math.max(0.03, 1/currentLayer.num_inputs) * MyRandom.getgaussian();
					weights.set(j, randWeight);
				}
			} else {
				int numWeights = currentLayer.num_inputs;
				weights = new Vec(numWeights);
				for(int j = 0; j < weights.size(); j++) {
					weights.set(j, 0.0);
				}
			}
			layerWeights.add(i, weights);
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
				x_row[p] = MyRandom.getgaussian();
			}
			X.takeRow(x_row);
		}
		
		Matrix Y = new Matrix(0, outputs);
		for(int n = 0; n < patterns; n++) {
			double[] y_row = new double[outputs];
			for(int p = 0; p < outputs; p++) {
				y_row[p] = MyRandom.getgaussian();
			}
			Y.takeRow(y_row);
		}
		
		// Generate weights with OLS			
		LayerLinear OLSlayer = new LayerLinear(inputs, outputs);
		Vec OLSweights = OLSlayer.ordinary_least_squares(X, Y);
		//System.out.println("OLS weights:");
		//System.out.println(OLSweights);
		
		// Generate weights with gradient descent
		NeuralNet GDtest = new NeuralNet(1.0, 1);
		LayerLinear GDlayer = new LayerLinear(inputs, outputs);
		GDtest.layerCollection.add(GDlayer);
		GDtest.initWeights();
		//System.out.println("Initial weights: ");
		//System.out.println(GDtest.layerWeights.get(0));
		// Set up gradient Vecs
		for(Layer layer : GDtest.layerCollection) {
			layer.gradients.fill(0.0);
			GDtest.layerGradients.add(layer.gradients);
		}
		for(int i = 0; i < 10000; i++) {
			GDtest.train(X, Y);
			/*
			int row_index = MyRandom.getinteger(patterns);
			Vec x_sample = X.row(row_index);
			Vec y_sample = Y.row(row_index);
			GDtest.refineWeights(x_sample, y_sample, 0.001);
			*/
		}
		Vec GDweights = GDtest.layerWeights.get(0);
		//System.out.println("GD weights: ");
		//System.out.println(GDweights);
		if(GDweights.squaredDistance(OLSweights) >= 0.5)
			throw new TestFailedException("testRefineWeights");
	}
		
	public static void testGradient() 
		throws TestFailedException {
			
		int outputs = 1;
		int inputs = 5;
		NeuralNet test = new NeuralNet(1.0, 1);
		LayerLinear linearlayerOne = new LayerLinear(inputs,3);
		test.layerCollection.add(linearlayerOne);	
		LayerLeaky tanhlayerOne = new LayerLeaky(3);
		test.layerCollection.add(tanhlayerOne);
		LayerLinear linearlayerTwo = new LayerLinear(3,2);
		test.layerCollection.add(linearlayerTwo);
		LayerTanh tanhlayerTwo = new LayerTanh(2);
		test.layerCollection.add(tanhlayerTwo);
		LayerLinear linearlayerThree = new LayerLinear(2, outputs);
		test.layerCollection.add(linearlayerThree);

		test.initWeights();
		/*
		System.out.println("Initial weights: ");
		System.out.println(test.layerWeights.get(0));
		System.out.println(test.layerWeights.get(1));
		System.out.println(test.layerWeights.get(2));
		System.out.println(test.layerWeights.get(3));
		System.out.println(test.layerWeights.get(4));
		*/
		
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
		//System.out.println("Target: ");
		//System.out.println(y);
		// Call predict() to generate activation
		test.predict(x);
		//System.out.println("Nonlinear activation: ");
		//System.out.println(tanhlayerOne.activation);
		// Call backProp() to generate blame
		test.backProp(test.layerWeights.get(2), y);
		//System.out.println("Nonlinear blame: ");
		//System.out.println(tanhlayerOne.blame);
		//System.out.println("Linear blame: " + linearlayerOne.blame);
		
		// Find gradient using both methods
		//System.out.println();
		//System.out.println("GradientFD method:");
		//System.out.println();
		test.updateGradientFD(x, y);
		for(int i = 0; i < test.layerCollection.size(); i++) {
			Vec gradientfd = test.layerGradients.get(i);
			//System.out.println("gradientfd");
			//System.out.println(gradientfd);
		}
		//System.out.println();
		//System.out.println("Gradient Descent method: ");
		//System.out.println();
		test.updateGradient(x);
		for(int i = 0; i < test.layerCollection.size(); i++) {
			Vec gradient = test.layerGradients.get(i);	
			//System.out.println("gradient");
			//System.out.println(gradient);
		}
		
		// Compare gradients
		//if(gradientfd.squaredDistance(gradient) >= Math.pow(10, -6))
			//throw new TestFailedException("testGradient");
	}
	 
	public String name() {
		return("My Neural Net");
	}		
}
	
	
