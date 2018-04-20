abstract class Layer {
	protected Vec activation;
	protected Vec blame;
	protected Vec gradients;
	protected int num_inputs, num_outputs, num_weights;
	protected int[] dim_inputs, dim_outputs, dim_filter;
	
	Layer(int inputs, int outputs) {
		
		activation = new Vec(outputs);
		blame = new Vec(outputs);
		gradients = new Vec((inputs + 1) * outputs);
		num_inputs = inputs;
		num_outputs = outputs;
		
	}
	
	Layer(int rows, int cols, int depth) {
		
		num_outputs = (rows * cols * depth) / 2;
		activation = new Vec(num_outputs);
		blame = new Vec(num_outputs);
		gradients = new Vec(0);
	}
	
	Layer(int _puts) {
		
		activation = new Vec(_puts);
		blame = new Vec(_puts);
		gradients = new Vec(0);
		num_inputs = _puts;
	
	}
	
	Layer(int[] inputDims, int[] filterDims, int[] outputDims) {
		// Set dimensions for tensors
		dim_inputs = inputDims;
		dim_outputs = outputDims;
		dim_filter = filterDims;
		
		// Set lengths for Vecs
		num_inputs = 1;
		for(int i = 0; i < dim_inputs.length; i++) num_inputs *= dim_inputs[i];
		num_outputs = 1;
		for(int i = 0; i < dim_outputs.length; i++) num_outputs *= dim_outputs[i];
		num_weights = 1;
		for(int i = 0; i < dim_filter.length; i++) num_weights *= dim_filter[i];
		num_weights += dim_filter[2];
		
		// Initialize Vecs for activation, blame, and gradients
		activation = new Vec(num_outputs);
		blame = new Vec(num_outputs);
		gradients = new Vec(num_weights);
	}
	
	abstract void activate(Vec weights, Vec x);
	abstract Vec backprop(Vec weights);
	abstract void updateGradient(Vec x, Vec gradient);
}
