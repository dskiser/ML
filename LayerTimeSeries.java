class LayerTimeSeries extends Layer {
	
	// First inputs - 1 units have a sine activation.  Last unit has the identity function as its activation.
	LayerTimeSeries(int inputs) {
		super(inputs);
	}
	
	public void activate(Vec weights, Vec x) {
		activation = new Vec(x.size());
		for(int i = 0; i < x.size()-1; i++) {
			activation.set(i, Math.sin(x.get(i)));
		}
		activation.set(x.size()-1, x.get(x.size()-1));
	}
	
	public Vec backprop(Vec weights) {
		Vec prevBlame = new Vec(blame.size());
		for(int i = 0; i < blame.size()-1; i++) {
			prevBlame.set(i, blame.get(i) * Math.cos(Math.asin(activation.get(i))));
		}
		prevBlame.set(blame.size()-1, blame.get(blame.size()-1));
		return prevBlame;
	}
	
	public void updateGradient(Vec x, Vec gradient) {}
}
