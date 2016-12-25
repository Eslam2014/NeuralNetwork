package fcis.asu.neural;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Layer {
	private Neural[] neurals;
	private double[] errors;
	private double[] outputs;
	private int nNeuralNextLayer;
	private int nNeural;

	private Layer(int nNeural, int nNeuralNextLayer, ActivationType activationType) {
		this.nNeural = nNeural;
		this.nNeuralNextLayer = nNeuralNextLayer;
		this.outputs = new double[nNeural];
		initalNeurals(nNeural, nNeuralNextLayer, activationType);
		errors = new double[nNeural];

	}

	public static Layer outputLayer(int nClasses, ActivationType activationType) {
		return new Layer(nClasses, 0, activationType);
	}

	public static Layer hiddenLayer(int nNeural, int nNeuralNextLayer, ActivationType activationType, double slope) {
		return new Layer(nNeural, nNeuralNextLayer, activationType);
	}

	public static Layer inputLayer(int nNeural, int nNeuralNextLayer, ActivationType activationType) {
		return new Layer(nNeural+1, nNeuralNextLayer, activationType);

	}

	private void initalNeurals(int nNeural, int nNeuralNextLayer, ActivationType activationType) {
		neurals = new Neural[nNeural];
		for (int i = 0; i < nNeural; i++) {
			neurals[i] = new Neural(nNeuralNextLayer, activationType);
		}
	}

	public double[][] getWeightForNextLayersSeq() {
		int nNeurals = this.neurals.length;
		double[][] weights = new double[nNeuralNextLayer][nNeurals];

		for (int i = 0; i < nNeuralNextLayer; i++) {
			for (int j = 0; j < nNeurals; j++) {
				weights[i][j] = neurals[j].getWeight(i);
			}
		}
		return weights;
	}

	public double[] calcOutputSeq(double[] input, double[][] inputWeights, double bias, ActivationType activationType,
			double slope) {
		for (int i = 0; i < nNeural; i++) {
			outputs[i] = neurals[i].calcOutput(input, inputWeights[i],activationType,slope);
		}
		return outputs;
	}

	public void setOutputForInputLayer(double[] output, double bias) {

		
		int lastIndex = nNeural - 1; // bias
		for (int i = 0; i < lastIndex; i++) {
			neurals[i].setOutput(output[i]);
			this.outputs[i] = output[i];
		}
		neurals[lastIndex].setOutput(bias);
		this.outputs[lastIndex] = bias;

	}

	public double[] clacOutputLayerError(int[] desire, double slope) {
		for (int i = 0; i < nNeural; i++) {
			errors[i] = neurals[i].calcOutputNeuralError(desire[i],slope);
		}
		return errors;
	}

	public double[] clacHiddenLayerError(double[] nextLayerError, double slope) {

		for (int i = 0; i < nNeural; i++) {
			errors[i] = neurals[i].calcHiddenNeuralError(nextLayerError,slope);
		}
		return errors;
	}

	public void updateWeights(double[] nextLayerError, double learningRate) {
		for (int i = 0; i < nNeural; i++) {
			neurals[i].updateWeight(nextLayerError, learningRate);
		}
	}

}
