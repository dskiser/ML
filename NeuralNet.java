import java.util.ArrayList;

class NeuralNet extends SupervisedLearner {
	
	protected double momentum, randWeight, new_weight, lambda1, lambda2;
	protected int batch_size, indexOutLayer, numLayers, numWeights, cycles, patterns, temp, swap_index, stop;
	protected int[] randindex;
	protected int count; // counts the patterns NN is trained on, in order to implement batch_size
	protected Vec prev_activation, final_activation, x, y, weights, current_weights, predicted, input, netgradients;
	protected boolean regularize = false;

	NeuralNet(double effMiniBatch, int miniBatch) {
		super();
	
		momentum = 1 - 1 / effMiniBatch;
		batch_size = miniBatch;
		count = 0;	
	}
	
	NeuralNet(double effMiniBatch, int miniBatch, double lambdaone, double lambdatwo) {
		super();
		
		momentum = 1 - 1 / effMiniBatch;
		batch_size = miniBatch;
		count = 0;
		
		lambda1 = lambdaone;
		lambda2 = lambdatwo;
		regularize = true;
	}
	
	public void regularize() {
		//System.out.println("regularlizing ..");
		//System.out.println(lambda1);
		Vec weights = layerWeights.get(layerWeights.size() - 1);
		weights.scale(1 - lambda2);
		weights.set(0, weights.get(0) * (1/(1-lambda2))); // don't regularize bias
		weights.set(weights.size()-1, weights.get(weights.size()-1) * 1/(1-lambda2)); // don't regularize weight from one hidden linear unit
		for(int i = 1; i < weights.size()-1; i++) { // don't regularize bias or weight from one hidden linear unit
			if(weights.get(i) >= 0) weights.set(i, weights.get(i) - lambda1);
			else weights.set(i, weights.get(i) + lambda1);
		}
		layerWeights.set(layerWeights.size() - 1, weights);
	}
	
	public Vec predict(Vec x) {
		prev_activation = x;
		for(int i = 0; i < layerCollection.size(); i++) {
			layerCollection.get(i).activate(layerWeights.get(i), prev_activation);
			prev_activation = layerCollection.get(i).activation;
			//System.out.println("Layer " + i + " activation:");
			//System.out.println(prev_activation);
		}
		final_activation = prev_activation;
		return final_activation;
	}
	
	public void train(Matrix X, Matrix Y) {
		//System.out.println("train in NN called");
		cycles = 1;
		for(int j = 0; j < cycles; j++) {
			// Create array of shuffled indexes
			patterns = X.rows();
			//System.out.println("number of patterns: " + patterns);
			randindex = new int[patterns];
			for(int i=0; i<randindex.length; i++) {
				randindex[i] = i;
			}
			for(int i=randindex.length-1; i>0; i--) {
				temp = randindex[i];
				swap_index = MyRandom.getinteger(i);
				randindex[i] = randindex[swap_index]; 
				randindex[swap_index] = temp;
			}
		
			
			stop = 0;
			
			for(int i : randindex) {
				x = X.row(i);
				//System.out.println(x);
				y = Y.row(i);
				
				this.refineWeights(x, y, 0.005);
				
				stop++;
				//if(stop == 1000) break; //only use 10000 patterns at a time
			}
		}
	}
	
	public Matrix train_unsupervised(Matrix X) {
		int channels = X.cols() / (64*48);
		//System.out.println("channels: " + channels);
		int n = X.rows();
		int k = 2;
		Matrix V = new Matrix(n, k);
		V.fill(0.0);
		double learning_rate = 0.1;
		
		int t, p, q, s;
		Vec features, labels, pred, input_blame;
		double[] vrow;
		for(int j = 0; j < 10; j++) {
			System.out.println("Beginning cycle " + (j+1) + ": ");
			for(int i = 0; i < 10000000; i++) {
				//System.out.println();
				//System.out.println("j=" + j + ", i=" + i);
				
				//int c = 0;
				/*
				for(Vec weights : layerWeights) {
					System.out.println("Layer" + (c++) + " weights=");
					System.out.println(weights);
				}
				*/
				t = MyRandom.getinteger(X.rows());
				p = MyRandom.getinteger(64);
				q = MyRandom.getinteger(48);
				/*
				t = i % 1000;
				p = (i * 31) % 64;
				q = (i * 19) % 48;
				*/
				//System.out.println("t=" + t + ", p=" + p + " , q="+ q);
				features = new Vec(new double[] { (double) p/64.0, (double) q/48.0 });
				features = features.attach(V.row(t));
				//System.out.println("in=" + features);
				s = channels * (64 * q + p);
				labels = new Vec(new double[] { X.get(t, s), X.get(t, s+1), X.get(t, s+2) });
				//System.out.println("target=" + labels);
				pred = predict(features);
				//System.out.println("prediction=" + pred);
				backProp(features, labels);
				/*
				c = 0;
				for(Layer layer : layerCollection) {
					System.out.println("Layer" + (c++) + "_blame=" + layer.blame);
				}
				*/
				
				// Compute blame term on inputs
				input_blame = layerCollection.get(0).backprop(layerWeights.get(0));
				//input_blame.scale(-1.0);
				//System.out.println(layerWeights.get(0));
				//System.out.println("v_gradinet=" + input_blame);
				updateGradient(features);
				for(int z = 0; z < layerCollection.size(); z++) {
					if(layerCollection.get(z) instanceof LayerLinear || layerCollection.get(z) instanceof LayerConv) {
						// Loop through each weight vector (and corresponding gradient vector)
						current_weights = layerWeights.get(z);
						for(int y = 0; y < current_weights.size(); y++) {
							new_weight = current_weights.get(y) + (layerGradients.get(z).get(y) * learning_rate);
							current_weights.set(y, new_weight);
						}
						layerWeights.set(z, current_weights);
					}
				}
				int w = 0;
				for(Layer layer : layerCollection) {
					if(momentum == 0.0) layer.gradients.fill(0.0);
					else layer.gradients.scale(momentum);
					layerGradients.set(w, layer.gradients);
					w++;
				}
				// Update inputs (only the ones indicating state)
				for(int h = 2; h < 4; h++) features.set(h, features.get(h) + input_blame.get(h)*learning_rate);
				vrow = new double[] { features.get(2), features.get(3) };
				//System.out.println("updated V[t] = " + vrow[0] + ", " + vrow[1]);
				V.setRow(t, vrow);
			}
			learning_rate *= 0.75;
		}
		return V;
	}
	
	public void refineWeights(Vec rowx, Vec rowy, double learning_rate) {
		// Add momentum
		int k = 0;
		for(Layer layer : layerCollection) {
			if(momentum == 0.0) layer.gradients.fill(0.0);
			else layer.gradients.scale(momentum);
			layerGradients.set(k, layer.gradients);
			k++;
		}
		
		weights = layerWeights.get(layerWeights.size()-1);
		this.predict(rowx);
		this.backProp(weights, rowy); // compute blame for each layer
		if(regularize) regularize();
		this.updateGradient(rowx); // compute gradients for each layer
		
		// Update weights in each layer
		if(count++ == batch_size) {
			count = 0;
			numLayers = layerCollection.size();
			//System.out.println(numLayers);
			for(int i = 0; i < numLayers; i++) {
				if(layerCollection.get(i) instanceof LayerLinear || layerCollection.get(i) instanceof LayerConv) {
					// Loop through each weight vector (and corresponding gradient vector)
					current_weights = layerWeights.get(i);
					//System.out.println(current_weights);
					for(int j = 0; j < current_weights.size(); j++) {
						new_weight = current_weights.get(j) + (layerGradients.get(i).get(j) * learning_rate);
						current_weights.set(j, new_weight);
					}
					layerWeights.set(i, current_weights);
					//if(i == 0) { System.out.println(current_weights); }
				}
			}
		}
	}
	
	// Compute blame for each layer in Neural Network
	public void backProp(Vec bpweights, Vec targets) {
		indexOutLayer = this.layerCollection.size()-1;
		predicted = this.layerCollection.get(indexOutLayer).activation;
		
		// Compute blame for output layer
		layerCollection.get(indexOutLayer).blame = targets.sum(-1, predicted);
		//System.out.println("Blame on output layer:");
		//System.out.println(layerCollection.get(indexOutLayer).blame);
		
		// Compute blame for each subsequent layer
		for(int i = indexOutLayer; i >= 1; i--) {
			//Vec current_weights = layerWeights.get(i);
			//Vec prevBlame = layerCollection.get(i).backprop(layerWeights.get(i));
			//System.out.println("Layer " + (i) + " blame: " + layerCollection.get(i).blame);
			layerCollection.get(i-1).blame = layerCollection.get(i).backprop(layerWeights.get(i)); 
		}
		//System.out.println("Layer 0 blame: " + layerCollection.get(0).blame);		
	}
	
	// Use activation of previous layer and blame of current layer to compute
	// gradient vector of each layer 
	public void updateGradient(Vec x) {
					
		// Update weights of first layer using original data
		layerCollection.get(0).updateGradient(x, layerWeights.get(0));
		layerGradients.set(0, layerCollection.get(0).gradients);
		
		// Update subsequent layer weights using activation from previous layers
		//Vec input, current_weights, netgradients;
		for(int i = 1; i < layerCollection.size(); i++) {
			// Input is activation from previous layer
			input = layerCollection.get(i-1).activation;
			current_weights = layerWeights.get(i);
			layerCollection.get(i).updateGradient(input, current_weights);
			netgradients = layerCollection.get(i).gradients;
			layerGradients.set(i, netgradients);
		}
	}
	
	public void updateGradientFD(Vec x, Vec targets) {
		//System.out.println("x: " + x);
		int numLayers = this.layerCollection.size();
		Vec prevActivation, weights, error, layerTarget;
		Layer layer;
		for(int h = 0; h < numLayers; h++) {
			if(h == 0)
				prevActivation = x;
			else prevActivation = this.layerCollection.get(h-1).activation;
			//System.out.println("prevActivation: " + prevActivation);
			layer = this.layerCollection.get(h);
			weights = new Vec(0);
			if(layer instanceof LayerLinear || layer instanceof LayerConv) // don't try to update max pooling weights
				weights = layerWeights.get(h);
			//System.out.println("Weights: " + weights);
			
			// Determine target for each layer
			error = layer.blame;
			//System.out.println("error: " + error);
			layerTarget = layer.activation.sum(1, layer.blame);
		
			// Loop through each weight and update gradient
			double[] gradient = new double[weights.size()];
			double stepSize = 1 * Math.pow(10, -6);
			Vec error1, error2;
			for(int i = 0; i < weights.size(); i++) {
				//System.out.println("Find gradient for weight " + i);
				// Step up
				weights.set(i, weights.get(i) + (0.5*stepSize));
				layer.activate(weights, prevActivation);
				//System.out.println("New activation: " + layer.activation);
				// Determine SSE of step up
				error1 = layerTarget.sum(-1, layer.activation);
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
				error2 = layerTarget.sum(-1, layer.activation);
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
			this.layerGradients.add(h, gradients);
		}	
	}
	
	public void initWeights() {
		numLayers = layerCollection.size();
		numWeights = 0;
		randWeight = 0.0;
		
		// Set up gradient and blame Vecs
		for(Layer layer : layerCollection) {
			layer.gradients.fill(0.0);
			layer.blame.fill(0.0);
			layerGradients.add(layer.gradients);
		}
		
		for(int i = 0; i < numLayers; i++) {
			//Layer currentLayer = layerCollection.get(i);
			
			if(layerCollection.get(i) instanceof LayerLinear) {
				
				numWeights = (layerCollection.get(i).num_inputs + 1) * layerCollection.get(i).num_outputs;
				weights = new Vec(numWeights);
				
				
				for(int j = 0; j < weights.size(); j++) {
					randWeight = Math.max(0.03, 1/layerCollection.get(i).num_inputs) * MyRandom.getgaussian();
					weights.set(j, randWeight);
				}
				/*
				for(int b = 0; b < layerCollection.get(i).num_outputs; b++) weights.set(b, 0.001*(double) b);
				int k = layerCollection.get(i).num_outputs;
				for(int r = 0; r < layerCollection.get(i).num_outputs; r++) {
					for(int c = 0; c < layerCollection.get(i).num_inputs; c++) {
						weights.set(k++, 0.007*(double)r + 0.003*(double)c);
					}
				}
				*/ 
			} else if(layerCollection.get(i) instanceof LayerConv) {
				numWeights = layerCollection.get(i).num_weights;
				weights = new Vec(numWeights);
				for(int j = 0; j < weights.size(); j++) {
					randWeight = MyRandom.getgaussian() / numWeights;
					weights.set(j, randWeight);
				}				
			} else {
				weights = new Vec(0);
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
		System.out.println("OLS weights:");
		System.out.println(OLSweights);
		
		// Generate weights with gradient descent
		NeuralNet GDtest = new NeuralNet(1.0, 1);
		LayerLinear GDlayer = new LayerLinear(inputs, outputs);
		GDtest.layerCollection.add(GDlayer);
		GDtest.initWeights();
		System.out.println("Initial weights: ");
		System.out.println(GDtest.layerWeights.get(0));
		// Set up gradient Vecs
		//for(Layer layer : GDtest.layerCollection) {
			//layer.gradients.fill(0.0);
			//GDtest.layerGradients.add(layer.gradients);
		//}
		for(int i = 0; i < 10000; i++) {
			GDtest.train(X, Y);
			
			int row_index = MyRandom.getinteger(patterns);
			Vec x_sample = X.row(row_index);
			Vec y_sample = Y.row(row_index);
			//GDtest.refineWeights(x_sample, y_sample, 0.001);
			
		}
		Vec GDweights = GDtest.layerWeights.get(0);
		System.out.println("GD weights: ");
		System.out.println(GDweights);
		if(GDweights.squaredDistance(OLSweights) >= 0.5)
			throw new TestFailedException("testRefineWeights");
	}
		
	public static void testGradient(boolean verbose) 
		throws TestFailedException {
			
		int outputs = 3;
		int inputs = 64;
		NeuralNet test = new NeuralNet(1.0, 1);
		

		LayerConv a = new LayerConv(new int[] {8, 8, 1}, new int[] {5, 5, 4}, new int[] {8, 8, 4});
		test.layerCollection.add(a);	
		LayerLeaky b = new LayerLeaky(8 * 8 * 4);
		test.layerCollection.add(b);
		LayerMaxPooling2D c = new LayerMaxPooling2D(8, 8, 4);
		test.layerCollection.add(c);
		LayerConv d = new LayerConv(new int[] {4, 4, 4, 1}, new int[] {3, 3, 4, 6}, new int[] {4, 4, 1, 6});
		test.layerCollection.add(d);
		LayerLeaky e = new LayerLeaky(4 * 4 * 6);
		test.layerCollection.add(e);
		LayerMaxPooling2D f = new LayerMaxPooling2D(4, 4, 6);
		test.layerCollection.add(f);
		LayerLinear g = new LayerLinear(2 * 2 * 6, 3);
		test.layerCollection.add(g);

		test.initWeights();
		
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
		test.backProp(test.layerWeights.get(2), y);
		
		// Find gradient using both methods
		Vec gradient = new Vec(0);
		Vec gradientfd = new Vec(0);
		test.updateGradientFD(x, y);
		for(int i = 0; i < test.layerCollection.size(); i++) {
			if(test.layerGradients.get(i).size() > 0) {
				gradientfd = gradientfd.attach(test.layerGradients.get(i));
			}
		}

		test.updateGradient(x);
		for(int i = 0; i < test.layerCollection.size(); i++) {
			if(test.layerGradients.get(i) != null && !(test.layerGradients.get(i).summation() == 0.0)) {
				gradient = gradient.attach(test.layerGradients.get(i));	
			}
		}
		if(verbose) {
			System.out.println("Empirical Gradients -- Computed Gradients");
			for(int i = 0; i < gradient.size(); i++)
				System.out.println(gradientfd.get(i) + " -- " + gradient.get(i));
		}
		
		// Compare gradients
		if(gradientfd.squaredDistance(gradient) >= Math.pow(10, -6)) {
			throw new TestFailedException("testGradient");
		} else { System.out.println(); System.out.println("Finite differencing test passed"); }
	}
	 
	public String name() {
		return("My Neural Net");
	}
	
	public static void testAssgn4(boolean verbose) 
		throws TestFailedException {
		
		///Debug spew
		NeuralNet debug = new NeuralNet(1.0, 1);
		debug.layerCollection.add(new LayerConv(new int[] {4,4,1}, new int[] {3, 3, 2}, new int[] {4,4,2}));
		debug.layerCollection.add(new LayerLeaky(4*4*2));
		debug.layerCollection.add(new LayerMaxPooling2D(4, 4, 2));
		
		debug.layerWeights.add(0, new Vec(new double[] {0.1, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19}));
		debug.layerWeights.add(1, new Vec(0));
		debug.layerWeights.add(2, new Vec(0));
		
		for(int i = 0; i < 3; i++)
			debug.layerGradients.add(new Vec(0));
		
		Vec x = new Vec(new double[] {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5});
		Vec target = new Vec(new double[] { 0.3, 0.2, 0.1, 0, 0.7, 0.6, 0.5, 0.4 });
		
		debug.predict(x);
		debug.backProp(debug.layerWeights.get(2), target);
		debug.updateGradient(x);
		// Update weights
		double learning_rate = 0.01;
		int numLayers = debug.layerCollection.size();
		for(int i = 0; i < numLayers; i++) {
			// Loop through each weight vector (and corresponding gradient vector)
			Vec current_weights = debug.layerWeights.get(i);
			if(debug.layerGradients.get(i) != null) {
				for(int j = 0; j < current_weights.size(); j++) {
					double new_weight = current_weights.get(j) + (debug.layerGradients.get(i).get(j) * learning_rate);
					current_weights.set(j, new_weight);
				}
			debug.layerWeights.set(i, current_weights);
			}
		}
		
		if(verbose) {
			// Print out activation
			for(int i = 0; i < debug.layerCollection.size(); i++) {
				System.out.println("Layer " + i + " activation: " + debug.layerCollection.get(i).activation);
				System.out.println();
			}
		
			// Print out blame
			for(int i = 0; i < debug.layerCollection.size(); i++) {
				System.out.println("Layer " + i + " blame: " + debug.layerCollection.get(i).blame);
				System.out.println();
			}
		
			// Print out target
			System.out.println("target: " + target);
			System.out.println();
		
			// Print out gradients
			for(int i = 0; i < debug.layerCollection.size(); i++) {
				System.out.println("Layer " + i + " gradients: " + debug.layerGradients.get(i));
				System.out.println();
			}
		
			// Print out updated weights
			for(int i = 0; i < debug.layerCollection.size(); i++) {
				if(debug.layerGradients.get(i) != null) {
					System.out.println("Layer " + i + " updated weights: " + debug.layerWeights.get(i));
					System.out.println();
				}
			}
		}
		
		Vec answer = new Vec(new double[] {0.058379999999999994,0.005379999999999999,0.009680000000000001,0.020218,0.030756,0.041832,0.05237,0.06290799999999999,0.07398400000000001,0.084522,0.09505999999999999,0.0964,0.102238,0.108076,0.10975200000000002,0.11559,0.12142800000000001,0.123104,0.128942,0.13478});

		if(!debug.layerWeights.get(0).equal(answer))
			throw new TestFailedException("testAssgn4");
	}
				
}
	
	
