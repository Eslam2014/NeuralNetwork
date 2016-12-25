package fcis.asu.neural;

import java.util.HashMap;
import java.util.Map;

import fcis.asu.utilities.Utilities;

/**
 * Generalize hebbian algorithem
 * 
 * @author Eslam Ali
 */
public class GHA {
	private int nPCA;
	private double learningRate;
	private double[][] weights;

	/**
	 * @param nPCA
	 *            desired number of principle component
	 * @param learningRate
	 */
	public GHA(int nPCA, int nFeatures, double learningRate) {
		this.nPCA = nPCA;
		this.learningRate = learningRate;
		this.weights = Utilities.generateRandomDouble2DArr(nPCA, nFeatures);
	}

	public Map<double[], ClassType> regenerateSamples(Map<double[], ClassType> samples) {
		Map<double[], ClassType> newSamples = new HashMap<double[], ClassType>();
		for (Map.Entry<double[], ClassType> sample : samples.entrySet()) {
			newSamples.put(calcPCAOutput(sample.getKey()), sample.getValue());
		}
		return newSamples;
	}

	public void trainPCAUseTestAndLearnSamples(Map<double[], ClassType> testSamples,
			Map<double[], ClassType> learnSamples, int nEpoch) {
		for (int i = 0; i < nEpoch; i++) {
			for (Map.Entry<double[], ClassType> sample : testSamples.entrySet()) {
				training(sample.getKey());
			} // end learn sample loop
			for (Map.Entry<double[], ClassType> sample : testSamples.entrySet()) {
				training(sample.getKey());
			} // end test sample loop
		} // end epoch loop

	}

	private void training(double[] features) {
		double[] tmpOutput = calcPCAOutput(features);
		weights = updateWeights(features, tmpOutput);
	}

	public double[] calcPCAOutput(double[] features) {
		double[] output = new double[nPCA];
		int nFeatures = features.length;

		for (int i = 0; i < nPCA; i++) {
			double tmpSum = 0;
			for (int j = 0; j < nFeatures; j++) {
				tmpSum += weights[i][j] * features[j];
			}
			output[i] = tmpSum;
		}
		return output;

	}

	private double[][] updateWeights(double[] features, double[] output) {
		int nFeatures = features.length;
		double[][] newWeights = new double[nPCA][nFeatures];
		for (int i = 0; i < nPCA; i++) {
			for (int j = 0; j < nFeatures; j++) {
				double tmpSum = 0;
				for (int k = 0; k <= i; k++) {
					tmpSum += (weights[i][k] * output[k]);
				}

				newWeights[i][j] = learningRate * ((features[j] * output[i]) - (output[i] * tmpSum));
			}
		}
		return newWeights;
	}

}
