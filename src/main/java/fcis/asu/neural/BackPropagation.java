package fcis.asu.neural;

import java.util.Map;

/**
 * Back-propagation algorithm Multilayer Perceptron (MLP) Network
 * 
 * @author Eslam Ali
 * 
 */
public class BackPropagation extends MLP {

	public BackPropagation(int nNeuralClasses, int nFeatures, int nHiddenLayer, int[] nNeuralInHiddenLayers,
			double learningRate, ActivationType activationType, double bias, double errorThreshold, double slope) {
		super(nNeuralClasses, nFeatures, nHiddenLayer, nNeuralInHiddenLayers, learningRate, activationType, bias,
				errorThreshold, slope);
	}

	@Override
	public void run(Map<double[], ClassType> learnSamples, Map<double[], ClassType> testSamles) {
		logger.info("MLP start runing...");
		super.run(learnSamples, testSamles);
	}

	@Override
	protected void learn(Map<double[], ClassType> learnSamples) {
		logger.info("MLP start learning...");
		double mse = Double.MAX_VALUE;
		while (mse > errorThreshold) {
			for (Map.Entry<double[], ClassType> sample : learnSamples.entrySet()) {
				inputLayer.setOutputForInputLayer(sample.getKey(), bias);
				double[] errorOfOutputLayer = forward(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(),
						sample.getValue().getDesire());
				backward(errorOfOutputLayer);
				feedforward();
			}
			mse = 0;
			for (Map.Entry<double[], ClassType> sample : learnSamples.entrySet()) {
				inputLayer.setOutputForInputLayer(sample.getKey(), bias);
				double[] errorOfOutputLayer = forward(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(),
						sample.getValue().getDesire());

				for (double err : errorOfOutputLayer)
					mse += (err * err) / 2;

			}
			mse /= learnSamples.size();
			System.out.println("mse=" + mse);

		}
		logger.info("MLP finished learning last mse:" + mse);

	}

	@Override
	protected void test(Map<double[], ClassType> testSamles) {

		logger.info("MLP start testing...");
		int counter = 0;
		for (Map.Entry<double[], ClassType> sample : testSamles.entrySet()) {
			inputLayer.setOutputForInputLayer(sample.getKey(), bias);
			forward(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(), sample.getValue().getDesire());
			double[] errorOfOutputLayer = outputLayer.getOutputs();
			int maxValueIndex = -1;
			double tmpMaxValue = Double.MIN_VALUE;
			int nClasses = errorOfOutputLayer.length;

			for (int i = 0; i < nClasses; i++) {
				if (errorOfOutputLayer[i] > tmpMaxValue) {
					tmpMaxValue = errorOfOutputLayer[i];
					maxValueIndex = i;
				}
			}

			if (sample.getValue().getDesire()[maxValueIndex] == 1) {
				counter++;
			}
		}
		testAccuracy = ((double) counter / testSamles.size()) * 100;
		logger.info("MLP finished testing with accuracy :" + testAccuracy);
	}

	private double[] forward(double[] input, double[][] inputWeights, int[] desireClass) {
		double[] tmpInput = input;
		double[][] tmpInputWeights = inputWeights;

		for (Layer hiddenLayer : hiddenLayers) {
			tmpInput = hiddenLayer.calcOutputSeq(tmpInput, tmpInputWeights, bias, activationType, slope);
			tmpInputWeights = hiddenLayer.getWeightForNextLayersSeq();

		}
		outputLayer.calcOutputSeq(tmpInput, tmpInputWeights, bias, activationType, slope);
		if (desireClass == null)
			return null;
		return outputLayer.clacOutputLayerError(desireClass, slope);
	}

	private void backward(double[] nextLayerError) {
		int nHiddenLayer = this.nHiddenLayer - 1;
		double[] tmpNextLayerError = nextLayerError;
		for (int i = nHiddenLayer; i >= 0; i--) {
			tmpNextLayerError = hiddenLayers[i].clacHiddenLayerError(tmpNextLayerError, slope);

		}
	}

	private void feedforward() {
		double[] tmpNextLayerError = nHiddenLayer > 0 ? hiddenLayers[0].getErrors() : outputLayer.getErrors();
		inputLayer.updateWeights(tmpNextLayerError, learningRate);
		for (int i = 1; i < nHiddenLayer - 1; i++) {
			tmpNextLayerError = hiddenLayers[i].getErrors();
			hiddenLayers[i - 1].updateWeights(tmpNextLayerError, learningRate);
		}

		if (nHiddenLayer > 0)
			hiddenLayers[nHiddenLayer - 1].updateWeights(outputLayer.getErrors(), learningRate);

	}

}
