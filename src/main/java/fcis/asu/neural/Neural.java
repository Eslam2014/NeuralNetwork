package fcis.asu.neural;

import fcis.asu.utilities.Utilities;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Neural {

	private double[] weights;
	private double error;
	private double output;
	private ActivationType activationType;
	

	public Neural(int nNeuralNextLaye, ActivationType activationType) {
		
		this.weights = nNeuralNextLaye > 0 ?Utilities.generateRandomDoubleArr(nNeuralNextLaye):null;
		this.activationType = activationType;
	}

	public double calcOutput(double[] inputs, double[] inputWeights,ActivationType activationType, double slope) {
		double net = Utilities.dotProduct(inputs, inputWeights);
		return this.output = activation(activationType, net, slope);
	}

	public double activation(ActivationType activation, double... args) {
		switch (activation) {
		case Sigmoid: {
			return Activation.sigmoid(args[0], args[1]);
		}
		case Tangent: {
			return Activation.tangentSigmoid(args[0], args[1]);
		}
		case Signum: {
			return Activation.signum(args[0]);
		}
		case Linear:{
			return args[0];
		}
		default:
			throw new IllegalArgumentException();

		}
	}

	public double calcHiddenNeuralError(double[] errors, double slope) {
		double sumError = Utilities.dotProduct(weights, errors);
		switch (activationType) {
		case Sigmoid:
			return this.error = slope * output * (1 - output) * sumError;
		case Tangent:
			throw new UnsupportedOperationException("Tangent not implemented yet");
		default:
			throw new UnsupportedOperationException("default not implemented yet");
		}

	}

	/**
	 * @param yk
	 *            result of activation function of last layer
	 * @param dk
	 *            desire output of this neural
	 * @return error
	 */
	public double calcOutputNeuralError(double dk, double slope) {
		switch (activationType) {
		case Sigmoid:
			return this.error = (double) slope * (dk - output) * output * (1 - output);
		case Tangent:
			throw new UnsupportedOperationException("Tangent not implemented yet");
		case Signum:
			return this.error = dk - output;
		case Linear:
			return this.error = dk - output;
		default:
			throw new UnsupportedOperationException("default not implemented yet");
		}

	}

	public void updateWeight(double[] nextLayerError, double learningRate) {
		int nWeights = this.weights.length;
		for (int i = 0; i < nWeights; i++) {
			weights[i] = weights[i] + (learningRate * nextLayerError[i] * output);
		}

	}

	public double getWeight(int index) {
		return this.weights[index];
	}

}
