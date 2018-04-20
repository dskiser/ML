class LayerMaxPooling2D extends Layer {
	
	protected double[] input, output, values;
	protected int[] indexes;
	protected int rows, cols, depth;
	protected int index, m, j;
	protected Vec blame_track, prevBlame;
	
	LayerMaxPooling2D(int row_num, int col_num, int depth_num) {
		super(row_num, col_num, depth_num);
		
		rows = row_num;
		cols = col_num;
		depth = depth_num;
		num_weights = rows * cols * depth;
		blame_track = new Vec(num_weights);
	}
	
	void activate(Vec weights, Vec x) {
		//System.out.println("Input size: " + x.size());
		blame_track.fill(0.0); // so that max values can be marked for calculating blame
		//System.out.println("weights in max pooling: " + weights.size());
		input = x.toArray();
		output = new double[input.length/4];
		
		values = new double[4];
		indexes = new int[4];
		index = 0;
		m = 0;
		for(int j = 0; j < depth; j++) {
			for(int i = 0; i < rows; i=i+2) {
				for(int k = 0; k < cols; k = k+2) {
					index = (j*rows*cols) + (i*cols) + k;
					
					values[0] = input[index];
					indexes[0] = index;
					values[1] = input[index+1];
					indexes[1] = index+1;
					values[2] = input[index + cols];
					indexes[2] = index + cols;
					values[3] = input[index + cols + 1];
					indexes[3] = index + cols + 1;
			
					output[m] = Math.max(Math.max(values[0], values[1]), Math.max(values[2], values[3]));
					//System.out.println("max: " + output[m]);
					
					for(int l = 0; l < values.length; l++) {
						//System.out.println("value: " + values[l] + " / index: " + indexes[l]);
						if(values[l] == output[m]) {
							blame_track.set(indexes[l], 1.0); // mark max values
							break;
						}
					}
					m++;
				}
			}
		}
		
		activation = new Vec(output);
		//System.out.println("Max layer activation size: " + activation.size());
	}
	
	Vec backprop(Vec weights) {
		prevBlame = new Vec(num_weights);
		 
		j = 0;
		for(int i = 0; i < blame_track.size(); i++) {
			if(blame_track.get(i) == 1.0) {
				prevBlame.set(i, this.blame.get(j));
				j++;
			} else { prevBlame.set(i, 0.0); }
		}
		
		return prevBlame;
	}
	
	void updateGradient(Vec x, Vec gradient) {}
}
