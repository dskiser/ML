class LayerTanh extends Layer {
	
	LayerTanh(int inputs) {
		super(inputs);
	}
	
	public void activate(Vec weights, Vec x) {
		activation = new Vec(x.size());
		for(int i = 0; i < x.size(); i++) {
			activation.set(i, Math.tanh(x.get(i)));
		}
	}
	
	public Vec backprop(Vec weights) {
		Vec prevBlame = new Vec(blame.size());
		for(int i = 0; i < blame.size(); i++) {
			prevBlame.set(i, blame.get(i) * (1.0 - (activation.get(i) * activation.get(i))));
		}
		return prevBlame;
	}
	
	public void updateGradient(Vec x, Vec gradient) {}
}
