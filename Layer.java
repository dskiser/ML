abstract class Layer {
	protected Vec activation;
	protected int num_inputs, num_outputs;
	
	Layer(int inputs, int outputs) {
		
		activation = new Vec(outputs);
		num_inputs = inputs;
		num_outputs = outputs;
	}
	
	abstract void activate(Vec weights, Vec x);
}
