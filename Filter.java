class Filter extends SupervisedLearner {
	
	protected NeuralNet NN;
	protected boolean categorize;
	protected boolean impute;
	protected boolean normalize;
	protected boolean initialize;
	protected Imputer impx, impy;
	protected Normalizer normx, normy;
	protected NomCat nomx, nomy;
	
	Filter(NeuralNet nn, boolean cat, boolean imp, boolean norm) {
		super();
		
		NN = nn;
		categorize = cat;
		impute = imp;
		normalize = norm;
		initialize = true;
	}
	
	public String name() {
		return("My Filter");
	}
	
	public void train(Matrix X, Matrix Y) {
		/*
		System.out.println("Before transformations: ");
		for(int i = 0; i < Y.rows(); i++) {
			System.out.print(Y.get(i, 0));
		}
		*/
		Matrix transformedX = X;
		Matrix transformedY = Y;
		if(normalize) {
			// normalize values in X
			normx = new Normalizer();
			normx.train(X);
			Matrix newX = normx.outputTemplate();
			int xrows = X.rows();
			for(int i = 0; i < xrows; i++) {
				double[] oldfeat = X.row(i).toArray();
				double[] newfeat = new double[newX.cols()];
				normx.transform(oldfeat, newfeat);
				newX.takeRow(newfeat);
			}
			transformedX = newX;
			// normalize values in Y
			normy = new Normalizer();
			normy.train(Y);
			Matrix newY = normy.outputTemplate();
			int yrows = Y.rows();
			for(int i = 0; i < yrows; i++) {
				double[] oldfeat = Y.row(i).toArray();
				double[] newfeat = new double[newY.cols()];
				normy.transform(oldfeat, newfeat);
				newY.takeRow(newfeat);
			}
			transformedY = newY;
		}
		/*
		System.out.println("After normalize: ");
		for(int i = 0; i < X.rows(); i++) {
			System.out.print(X.get(i, 0));
		}
		*/
		if(impute) {
			// impute values in X
			impx = new Imputer();
			impx.train(X);
			Matrix newX = impx.outputTemplate();
			int xrows = X.rows();
			for(int i = 0; i < xrows; i++) {
				double[] oldfeat = X.row(i).toArray();
				double[] newfeat = new double[newX.cols()];
				impx.transform(oldfeat, newfeat);
				newX.takeRow(newfeat);
			}
			transformedX = newX;
			// impute values in Y
			impy = new Imputer();
			impy.train(Y);
			Matrix newY = impy.outputTemplate();
			int yrows = Y.rows();
			for(int i = 0; i < yrows; i++) {
				double[] oldfeat = Y.row(i).toArray();
				double[] newfeat = new double[newY.cols()];
				impy.transform(oldfeat, newfeat);
				newY.takeRow(newfeat);
			}
			transformedY = newY;
		}
		/*
		System.out.println("After impute: ");
		for(int i = 0; i < X.rows(); i++) {
			System.out.print(X.get(i, 0));
		}
		*/
		
		if(categorize) {
			// transform X
			nomx = new NomCat();
			nomx.train(X);
			Matrix newX = nomx.outputTemplate();
			int xrows = X.rows();
			for(int i = 0; i < xrows; i++) {
				double[] oldfeat = X.row(i).toArray();
				double[] newfeat = new double[newX.cols()];
				nomx.transform(oldfeat, newfeat);
				newX.takeRow(newfeat);
			}
			transformedX = newX;
			// transform Y
			nomy = new NomCat();
			nomy.train(Y);
			Matrix newY = nomy.outputTemplate();
			int yrows = Y.rows();
			for(int i = 0; i < yrows; i++) {
				double[] oldlabel = Y.row(i).toArray();
				double[] newlabel = new double[newY.cols()];
				nomy.transform(oldlabel, newlabel);
				newY.takeRow(newlabel);
			}
			transformedY = newY;
		}
		/*
		System.out.println("After categorize: ");
		for(int i = 0; i < X.rows(); i++) {
			System.out.print(X.get(i, 0));
		}
		System.out.println("Transformed matrix X: ");
		System.out.println(X);
		System.out.println("inputs: " + X.cols() + " outputs: " + Y.cols());
		*/
		if(!initialize) {
			//System.out.println("Transformed matrix Y: ");
			//System.out.println(transformedY);
			NN.train(transformedX, transformedY);
			//for(Vec weights : NN.layerWeights)
				//System.out.println("weights: " + weights);
		}
	}
	
	Vec predict(Vec x) {
		// Transform x
		double[] input = x.toArray();
		if(normalize) { // do normalization before imputation, since unknown columns imputed as 0.0
			int new_columns = normx.outputTemplate().cols();
			double[] newinput = new double[new_columns];
			normx.transform(input, newinput);
			input = newinput;
			/*
			System.out.print("normalize: ");
			for(int i = 0; i < newinput.length; i++)
				System.out.print(newinput[i]);
			System.out.println();
			*/
		} 
		if(impute) {
			int new_columns = impx.outputTemplate().cols();
			double[] newinput = new double[new_columns];
			impx.transform(input, newinput);
			input = newinput;
			/*
			System.out.print("impute: ");
			for(int i = 0; i < newinput.length; i++)
				System.out.print(newinput[i]);
			System.out.println();
			*/
		}
		if(categorize) {
			int new_columns = nomx.outputTemplate().cols();
			double[] newinput = new double[new_columns];
			nomx.transform(input, newinput);
			input = newinput;
			/*
			System.out.print("categorize: ");
			for(int i = 0; i < newinput.length; i++)
				System.out.print(newinput[i]);
			System.out.println();
			*/
		}
		Vec inputVec = new Vec(input);
		//System.out.println("Vec to be predicted on: " + inputVec);
		
		// Predict
		Vec activation = NN.predict(inputVec);
		//System.out.println("Activation returned: " + activation);
		
		// Untransform activation (don't need impute)
		double[] output = activation.toArray();
		if(categorize) {
			double[] new_output = new double[nomy.origColumns()];
			nomy.untransform(output, new_output);
			output = new_output;
		}
		if(normalize) {
			double[] new_output = new double[output.length];
			normy.untransform(output, new_output);
			output = new_output;
		} 
		Vec outputVec = new Vec(output);
		//System.out.println(outputVec + " ");
		return outputVec;
	}
}

