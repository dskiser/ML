class LayerTanh extends Layer {
	protected Vec activation;
	protected Vec blame;
	protected int num_inputs;
	
	LayerTanh(int inputs) {
		super(inputs);
	}
	
	public void activate(Vec weights, Vec x) {
		for(int i = 0; i < activation.size(); i++) {
			activation.set(i, Math.tanh(x.get(i)));
		}
	}
	
	public void backprop(Vec weights, Vec prevBlame) {
		for(int i = 0; i < blame.size(); i++) {
			blame.set(i, 1.0 - (activation.get(i) * activation.get(i));
		}
	}
	
	public Vec updateGradient(Vec x, Vec gradient) {
		Vec na = new Vec(0);
		return na;
	}
}
