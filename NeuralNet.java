import java.util.ArrayList;

class NeuralNet extends SupervisedLearner {
	protected ArrayList<Layer> layerCollection;
	protected Vec layerWeights;
	
	NeuralNet(ArrayList<Layer> layerObjects, Vec weights) {
		layerCollection = layerObjects;
		layerWeights = weights;
	}
	
	void predict(Vec inputs) {
		layerCollection(0).activate(layerWeights, inputs);
	}
	
	void train(Matrix X, Matrix Y) {
		layerCollection(0).ordinary_least_squares(X, Y);
	}
}
	
	
