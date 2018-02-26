class LayerLeaky extends Layer {
	
	LayerLeaky(int inputs) {
		super(inputs);
	}
	
	public void activate(Vec weights, Vec x) {
		activation = new Vec(x.size());
		for(int i = 0; i < x.size(); i++) {
			if(x.get(i) < 0) activation.set(i, 0.01 * x.get(i));
			else activation.set(i, x.get(i));
		}
	}
	
	public Vec backprop(Vec weights) {
		Vec prevBlame = new Vec(blame.size());
		for(int i = 0; i < blame.size(); i++) {
			if(activation.get(i) < 0) prevBlame.set(i, blame.get(i) * 0.01);
			else prevBlame.set(i, blame.get(i));
		}
		return prevBlame;
	}
	
	public void updateGradient(Vec x, Vec gradient) {}
}
