package fcis.asu.neural;

import java.util.Map;

import org.apache.log4j.Logger;

/**
 * Resenblatt’s perceptron Single Layer Network
 * 
 * @author Eslam Ali
 *
 */
public class SLP extends NeuralNetwork {

	final static Logger logger = Logger.getLogger(SLP.class);
	private int maxNLearnEpoch;

	public SLP(int nNeuralClasses, int nFeatures, double learningRate, double bias, int maxNEpoch) {
		super(nNeuralClasses, nFeatures, learningRate, ActivationType.Signum, bias);
		this.inputLayer = Layer.inputLayer(nFeatures, 1, activationType);
		this.maxNLearnEpoch = maxNEpoch;

	}

	@Override
	public void run(Map<double[], ClassType> learnSamples, Map<double[], ClassType> testSamles) {
		logger.info("start run SLP");
		super.run(learnSamples, testSamles);
	}

	@Override
	protected void learn(Map<double[], ClassType> learnSamples) {
		logger.info("SLP start learning");
		int nSamples = learnSamples.size();
		int correct = 0;
		for (int i = 0; i < maxNLearnEpoch; i++) {
			correct = 0;
			for (Map.Entry<double[], ClassType> sample : learnSamples.entrySet()) {
				inputLayer.setOutputForInputLayer(sample.getKey(), bias);
				outputLayer.calcOutputSeq(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(), bias,
						activationType, 1);
				double[] error = outputLayer.clacOutputLayerError(sample.getValue().getDesire(), 1);
				if (error[0] == 0) {
					correct++;
				}
				inputLayer.updateWeights(error, learningRate);
			} // end map samples loop
			if (nSamples == correct) {
				break;
			}
		} // end maxNLearnEpoch loop
		learnAccuracy = ((double) correct / nSamples) * 100;
		logger.info("SLP finished learn neural with accuracy:" + learnAccuracy);

	}

	@Override
	protected void test(Map<double[], ClassType> testSamles) {
		logger.info("SLP start testing neural network");
		int correct = 0;
		for (Map.Entry<double[], ClassType> sample : testSamles.entrySet()) {
			inputLayer.setOutputForInputLayer(sample.getKey(), bias);
			outputLayer.calcOutputSeq(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(), bias,
					activationType, 1);
			double[] error = outputLayer.clacOutputLayerError(sample.getValue().getDesire(), 1);
			if (error[0] == 0)
				correct++;
		} // end map samples loop
		this.testAccuracy = ((double) correct / testSamles.size()) * 100;
		logger.info("SLP finished test neural with accuracy:" + testAccuracy);

	}

}
