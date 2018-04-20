class Filter extends SupervisedLearner {
	
	protected NeuralNet NN;
	protected boolean catX, catY, normX, normY, impX, impY, initialize;
	protected Imputer impx, impy;
	protected Normalizer normx, normy;
	protected NomCat nomx, nomy;
	protected Matrix transformedX, transformedY, newX, newY;
	protected double[] oldfeat, newfeat, oldlabel, newlabel, newrow;
	protected double[] input, newinput, output, new_output;
	protected int xrows, yrows, new_columns;
	protected Vec inputVec, activation, outputVec;
	
	Filter(NeuralNet nn, boolean categorizeX, boolean imputeX, boolean normalizeX,
							boolean categorizeY, boolean imputeY, boolean normalizeY) {
		super();
		
		NN = nn;
		catX = categorizeX;
		impX = imputeX;
		normX = normalizeX;
		catY = categorizeY;
		impY = imputeY;
		normY = normalizeY;
		
		initialize = true;
	}
	
	public String name() {
		return("My Filter");
	}
	
	public void train(Matrix X, Matrix Y) {
		
		if(normX) {
			// normalize values in X
			normx = new Normalizer();
			normx.train(X);
			newX = normx.outputTemplate();
			xrows = X.rows();
			for(int i = 0; i < xrows; i++) {
				oldfeat = X.removeRow(i);
				newfeat = new double[newX.cols()];
				normx.transform(oldfeat, newfeat);
				X.insertRow(i, newfeat);
			}
		}
		
		if(normY) {
			// normalize values in Y
			normy = new Normalizer();
			normy.train(Y);
			newY = normy.outputTemplate();
			yrows = Y.rows();
			for(int i = 0; i < yrows; i++) {
				oldlabel = Y.removeRow(i);
				newlabel = new double[newY.cols()];
				normy.transform(oldlabel, newlabel);
				Y.insertRow(i, newlabel);
			}
		}
		
		if(impX) {
			// impute values in X
			impx = new Imputer();
			impx.train(X);
			newX = impx.outputTemplate();
			xrows = X.rows();
			for(int i = 0; i < xrows; i++) {
				oldfeat = X.row(i).toArray();
				newfeat = new double[newX.cols()];
				impx.transform(oldfeat, newfeat);
				newX.takeRow(newfeat);
			}
			X = newX;
		}
		if(impY) {
			// impute values in Y
			impy = new Imputer();
			impy.train(Y);
			newY = impy.outputTemplate();
			yrows = Y.rows();
			for(int i = 0; i < yrows; i++) {
				oldfeat = Y.row(i).toArray();
				newfeat = new double[newY.cols()];
				impy.transform(oldfeat, newfeat);
				newY.takeRow(newfeat);
			}
			Y = newY;
		}
		
		if(catX) {
			// transform X
			nomx = new NomCat();
			nomx.train(X);
			newX = nomx.outputTemplate();
			xrows = X.rows();
			for(int i = 0; i < xrows; i++) {
				
				oldfeat = X.removeRow(i);
				newfeat = new double[newX.cols()];
				nomx.transform(oldfeat, newfeat);
				X.insertRow(i, newfeat);
			}
		}
		if(catY) {
			
			// transform Y
			nomy = new NomCat();
			nomy.train(Y);
			newY = nomy.outputTemplate();
			yrows = Y.rows();
			for(int i = 0; i < yrows; i++) {
				
				oldlabel = Y.row(i).toArray();
				newlabel = new double[newY.cols()];
				nomy.transform(oldlabel, newlabel);
				newY.insertRow(i, newlabel);
				
			}
			Y = newY;
		}
		
		if(!initialize) {
			NN.train(X, Y);
		}
	}
	
	Vec predict(Vec x) {
		// Transform x
		input = x.toArray();
		if(normX) { // do normalization before imputation, since unknown columns imputed as 0.0
			new_columns = normx.outputTemplate().cols();
			newinput = new double[new_columns];
			normx.transform(input, newinput);
			input = newinput;
		} 
		if(impX) {
			new_columns = impx.outputTemplate().cols();
			newinput = new double[new_columns];
			impx.transform(input, newinput);
			input = newinput;
		}
		if(catX) {
			new_columns = nomx.outputTemplate().cols();
			newinput = new double[new_columns];
			nomx.transform(input, newinput);
			input = newinput;
		}
		inputVec = new Vec(input);
		
		// Predict
		activation = NN.predict(inputVec);
		
		// Untransform activation (don't need impute)
		output = activation.toArray();
		if(catY) {
			new_output = new double[nomy.origColumns()];
			nomy.untransform(output, new_output);
			output = new_output;
		}
		if(normY) {
			new_output = new double[output.length];
			normy.untransform(output, new_output);
			output = new_output;
		} 
		outputVec = new Vec(output);
		return outputVec;
	}
}

