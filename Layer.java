abstract class Layer {
	protected Vec activation;
	protected Vec blame;
	protected Vec gradients;
	protected int num_inputs, num_outputs;
	
	Layer(int inputs, int outputs) {
		
		activation = new Vec(outputs);
		blame = new Vec(outputs);
		gradients = new Vec((inputs + 1) * outputs);
		num_inputs = inputs;
		num_outputs = outputs;
		
	}
	
	Layer(int inputs) {
		
		activation = new Vec(inputs);
		blame = new Vec(inputs);
		gradients = new Vec(inputs);
		num_inputs = inputs;
	
	}
	
	abstract void activate(Vec weights, Vec x);
	abstract Vec backprop(Vec weights);
	abstract void updateGradient(Vec x, Vec gradient);
}
