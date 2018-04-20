import java.util.ArrayList;

class LayerConv extends Layer {
	protected Vec m, b, b_gradients, outputs, assoc_blame, prevBlame;
	protected Tensor input, filter, output, blame_tensor, prevBlame_tensor, gradient_tensor;
	int outputs_per_blame, filter_layer_size, m_weights;
	int[] individual_filter_size;
	double b_blame;
	//ArrayList<Tensor> filters;
	
	LayerConv(int[] inputDims, int[] filterDims, int[] outputDims) {
		super(inputDims, filterDims, outputDims);
		
		//filters = new ArrayList<Tensor>(1);
		
	}
	
	void activate(Vec weights, Vec x) {
		// Divide up filter weights
		b = new Vec(weights, 0, dim_filter[2]);
		m = new Vec(weights, dim_filter[2], weights.size() - dim_filter[2]);
		m_weights = dim_filter[1] * dim_filter[0];
		individual_filter_size = new int[] {dim_filter[0], dim_filter[1]};
		filter = new Tensor(m, dim_filter);
		//for(int i = 0; i < b.size(); i++) {
			//filters.add(new Tensor(new Vec(weights, b.size() + m_weights*i, m_weights), individual_filter_size));
		//}
		
		// Build tensors
		input = new Tensor(x, dim_inputs);
		outputs = new Vec(num_outputs);
		//int[] new_dim_outputs = {dim_outputs[0], dim_outputs[1]};
		output = new Tensor(outputs, dim_outputs);
		
		// Convolve
		
		//for(int i = 0; i < b.size(); i++) {
			Tensor.convolve(input, filter, output, false, 1);
			//System.out.println("LayerConv output size: " + output.size());
			// Add bias
			filter_layer_size = num_outputs / b.size();
			for(int i = 0; i < b.size(); i++) {
				for(int j = 0; j < filter_layer_size; j++) {
					output.set((i*filter_layer_size) + j, output.get((i*filter_layer_size)+j) + b.get(i));
				}
			}
			activation = output;
			//System.out.println("LayerConv activation size: " + activation.size());
			//activation.attach(output);
		//}
	}
	int outputs_per_bias;
	Vec reordered_blame, segment;
	Vec backprop(Vec weights) {
		// Divide up filter weights
		b = new Vec(weights, 0, dim_filter[2]);
		m = new Vec(weights, dim_filter[2], weights.size() - dim_filter[2]);
		// Build tensors
		//outputs_per_bias = m.size() / b.size();
		//reordered_m = new Vec(0);
		//for(int i = b.size() - 1; i >=0; i--) {
			//segment = new Vec(m, i*outputs_per_bias, outputs_per_bias); 
			//reordered_m = reordered_m.attach(segment);
		//}
		//reordered_m.reverse();
		filter = new Tensor(m, dim_filter);
		//this.blame.reverse();
		blame_tensor = new Tensor(this.blame, dim_outputs);
		prevBlame = new Vec(num_inputs);
		prevBlame_tensor = new Tensor(prevBlame, dim_inputs);
		
		// Convolve
		Tensor.convolve(filter, blame_tensor, prevBlame_tensor, true, 1);
		prevBlame = prevBlame_tensor;
		outputs_per_bias = prevBlame.size() / b.size();
		reordered_blame = new Vec(0);
		for(int i = b.size() - 1; i >=0; i--) {
			segment = new Vec(prevBlame, i*outputs_per_bias, outputs_per_bias); 
			reordered_blame = reordered_blame.attach(segment);
		}
		//prevBlame.reverse();
		//System.out.println("reordered blame: " + reordered_blame);
		//System.out.println("prevBlame: " + prevBlame);
		return prevBlame; 
	}
	
	void updateGradient(Vec x, Vec gradient) {
		// Divide up gradient weights
		b = new Vec(gradients, 0, dim_filter[2]);
		m = new Vec(gradients, dim_filter[2], gradients.size() - dim_filter[2]);
		// Build tensors
		gradient_tensor = new Tensor(m, dim_filter);
		
		input = new Tensor(x, dim_inputs);
		blame_tensor = new Tensor(this.blame, dim_outputs);
		
		// Convolve
		Tensor.convolve(input, blame_tensor, gradient_tensor, false, 1);
		//gradients = gradient_tensor;
		
		// Calculate blame on biases
		//System.out.println("b length: " + b.size());
		//System.out.println("blame length: " + blame.size());
		b_gradients = new Vec(b.size());
		outputs_per_blame = blame.size() / b.size();
		for(int i = b.size()-1; i >= 0; i--) {
			assoc_blame = new Vec(this.blame, i * outputs_per_blame, outputs_per_blame);
			b_blame = assoc_blame.summation();
			b_gradients.set(i, b_blame);
		} // bias gradients are backwards relative to other weights
		//System.out.println("b_gradients: " + b_gradients);
		gradients = b_gradients.attach(gradient_tensor);		
	}
}

