import java.util.ArrayList;

class NeuralNet extends SupervisedLearner {

	NeuralNet() {
		super();
	}
	
	public Vec predict(Vec x) {
		layerCollection.get(0).activate(layerWeights.get(0), x);
		Vec y_hat = layerCollection.get(0).activation;
		return y_hat;
	}
	
	public void train(Matrix X, Matrix Y) {
		Vec weights = layerCollection.get(0).ordinary_least_squares(X, Y);
		layerWeights.add(0, weights);
	}
	
	
	public String name() {
		return("Ordinary Least Squares");
	}		
}
	
	
