abstract class Layer {
	protected Vec activation;
	protected Vec blame;
	protected int num_inputs, num_outputs;
	
	Layer(int inputs, int outputs) {
		
		activation = new Vec(outputs);
		blame = new Vec(outputs);
		num_inputs = inputs;
		num_outputs = outputs;
		
	}
	
	abstract void activate(Vec weights, Vec x);
	abstract void backprop(Vec weights, Vec prevBlame);
	abstract Vec updateGradient(Vec x, Vec gradient);
}
