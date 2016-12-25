package fcis.asu.neural;

import java.util.Map;

import org.apache.log4j.Logger;

/**
 * Least Mean Square algorithm for Single Layer Network
 * 
 * @author Eslam Ali
 *
 */
public class LeastMeanSquare extends NeuralNetwork {

	final static Logger logger = Logger.getLogger(LeastMeanSquare.class);
	double stopMse;

	public LeastMeanSquare(int nNeuralClasses, int nFeatures, double learningRate, double bias, double stopMse) {
		super(nNeuralClasses, nFeatures, learningRate, ActivationType.Linear, bias);
		this.stopMse = stopMse;
		inputLayer = Layer.inputLayer(nFeatures, 1, activationType);
	}

	@Override
	protected void learn(Map<double[], ClassType> learnSamples) {
		logger.info("LMS start learning neural");
		double mse = Double.MAX_VALUE;
		int apoch = 0;
		int nSample = learnSamples.size();
		while (mse > stopMse) {

			for (Map.Entry<double[], ClassType> sample : learnSamples.entrySet()) {
				inputLayer.setOutputForInputLayer(sample.getKey(), bias);
				outputLayer.calcOutputSeq(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(), bias,
						activationType, 1);
				double[] error = outputLayer.clacOutputLayerError(sample.getValue().getDesire(), 1);
				inputLayer.updateWeights(error, learningRate);
			} // end map samples loop

			mse = 0;
			for (Map.Entry<double[], ClassType> sample : learnSamples.entrySet()) {
				inputLayer.setOutputForInputLayer(sample.getKey(), bias);
				outputLayer.calcOutputSeq(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(), bias,
						activationType, 1);
				double[] error = outputLayer.clacOutputLayerError(sample.getValue().getDesire(), 1);
				mse += (error[0] * error[0]) / 2;
			}
			mse /= nSample;
			++apoch;
		} // end while loop
			// learnAccuracy = ((double) correct / nSamples) * 100;
		logger.info("LMS finished learning final mse :" + mse + "after:" + apoch + " apoch");

	}

	@Override
	protected void test(Map<double[], ClassType> testSamles) {
		logger.info("LMS start test neural");
		int correct = 0;
		for (Map.Entry<double[], ClassType> sample : testSamles.entrySet()) {
			inputLayer.setOutputForInputLayer(sample.getKey(), bias);
			outputLayer.calcOutputSeq(inputLayer.getOutputs(), inputLayer.getWeightForNextLayersSeq(), bias,
					ActivationType.Signum, 1);
			double[] error = outputLayer.clacOutputLayerError(sample.getValue().getDesire(), 1);
			if (error[0] == 0)
				correct++;

		} // end map samples loop
		testAccuracy = ((double) correct / testSamles.size()) * 100;
		logger.info("LMS finish test neural with accuracy:" + testAccuracy);

	}

}
