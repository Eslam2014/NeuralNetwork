package fcis.asu.neural;

import java.util.Map;

import lombok.Getter;
import lombok.NonNull;


public abstract class NeuralNetwork {
	@Getter
	protected double learningRate;
	@Getter
	protected ActivationType activationType;
	@Getter protected double bias;
	protected Layer inputLayer;
	protected Layer outputLayer;
	@Getter protected int nNeuralClasses;
	@Getter protected int nFeaturs;
	@Getter protected double testAccuracy;
	@Getter protected double learnAccuracy;

	public NeuralNetwork(int nNeuralClasses,int nFeatures,double learningRate,@NonNull ActivationType activationType,double bias) {
		if(nNeuralClasses <= 0)
			throw new IllegalArgumentException("number of neural class must be more than 0");
		if(nFeatures <= 0)
			throw new IllegalArgumentException("nFeatures must be more than 0");
		
		this.learningRate=learningRate;
		this.activationType=activationType;
		this.bias=bias;
		this.nNeuralClasses=nNeuralClasses;
		this.outputLayer=Layer.outputLayer(nNeuralClasses, activationType);
		this.testAccuracy=0.0;
		this.learnAccuracy=0.0;
	}
	
	public void run(Map<double[], ClassType> learnSamples, Map<double[], ClassType> testSamles){
		if(learnSamples== null || testSamles == null)
			throw new IllegalArgumentException("samples must be not null");
		learn(learnSamples);
		test(testSamles);
	}
	
	protected abstract void learn(Map<double[], ClassType> learnSamples);
	protected abstract void test(Map<double[], ClassType> testSamles);

}
