package fcis.asu.neural;

import org.apache.log4j.Logger;

import lombok.Getter;

public abstract class MLP extends NeuralNetwork {

	@Getter
	protected Layer[] hiddenLayers;
	@Getter
	protected double errorThreshold;
	@Getter
	protected int nHiddenLayer;
	@Getter
	protected double slope;

	final static Logger logger = Logger.getLogger(MLP.class);
	
	public MLP(int nNeuralClasses, int nFeatures, int nHiddenLayer, int[] nNeuralInHiddenLayers, double learningRate,
			ActivationType activationType, double bias, double errorThreshold, double slope) {
		super(nNeuralClasses, nFeatures, learningRate, activationType, bias);
		this.errorThreshold = errorThreshold;
		this.nHiddenLayer = nHiddenLayer;
		this.slope = slope;
		inputLayer = Layer.inputLayer(nFeatures, nHiddenLayer > 0 ? nNeuralInHiddenLayers[0] : 1, activationType);
		initalHiddenLayers(nHiddenLayer, nNeuralInHiddenLayers, nNeuralClasses);
	}

	
	private void initalHiddenLayers(int nHidenLayer, int[] nNeuralInHidenLayers, int nClasses) {
		this.hiddenLayers = new Layer[nHidenLayer];
		// last hidden layer weights we be based on n of neural classes
		this.hiddenLayers[nHidenLayer - 1] = Layer.hiddenLayer(nNeuralInHidenLayers[nHidenLayer - 1], nClasses,
				activationType, slope);
		int tmp = nHidenLayer - 1;
		for (int i = 0; i < tmp; i++) {
			this.hiddenLayers[i] = Layer.hiddenLayer(nNeuralInHidenLayers[i], nNeuralInHidenLayers[i + 1],
					activationType, slope);
		}
	}

}
