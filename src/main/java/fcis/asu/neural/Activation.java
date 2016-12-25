
package fcis.asu.neural;

public class Activation {

	/**
	 * the neuron will has output signal only if its activation potential is
	 * non-negative, a property known as all-or-none
	 * 
	 * @param vk
	 *            is the activation potential of neuron k
	 * @return Yk
	 */
	public static int threshold(double vk) {
		return vk >= 0 ? 1 : 0;
	}

	/**
	 * @param vk
	 *            is the activation potential of neuron k
	 * @return Yk
	 */
	public static double piecewiseLinear(double vk) {
		if (vk >= .5)
			return 1;
		else if (-.5 < vk && vk < .5)
			return vk + .5;
		else
			return 0;
	}

	/**
	 * @param vk
	 *            is the activation potential of neuron k
	 * @param a
	 *            is the slop parameter of the sigmoid function.
	 * @return Yk
	 */
	public static double sigmoid(double vk, double a) {
		return ((double) 1 / (1 + Math.exp(-a*vk)));
	}

	/**
	 * @param vk
	 *            is the activation potential of neuron k
	 * @param a
	 *            is the slop parameter of the sigmoid function.
	 * @return Yk
	 */
	public static double tangentSigmoid(double vk, double a) {
		return ((1 - Math.exp(-a * vk)) / (1 + Math.exp(-a * vk)));
	}

	public static double signum(double y) {
		return y >= 0 ? 1 : -1;
	}

	public static double sigmoidDerivative(double yk, double dk, double slope) {
		return slope * (dk - yk) * yk * (1 - yk);
	}

	public static double tangentDerivative(double yk, double dk, double slope) {
		throw new UnsupportedOperationException();

	}
}
