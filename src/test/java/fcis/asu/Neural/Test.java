package fcis.asu.Neural;

import fcis.asu.neural.ActivationType;
import fcis.asu.neural.BackPropagation;
import fcis.asu.neural.GHA;
import fcis.asu.neural.LeastMeanSquare;
import fcis.asu.neural.NeuralNetwork;
import fcis.asu.neural.SLP;
import fcis.asu.utilities.FileProperties;
import fcis.asu.utilities.ReadSamples;

public class Test {
	public static void main(String[] args){
			
		FileProperties prop = FileProperties.builder().
				filePath("src/main/resources/IrisData.txt")
				.targetClass("Iris-setosa")
				//.targetClass("Iris-versicolor")
				.targetClass("Iris-virginica")
				.targetFeature(0)
				.targetFeature(1)
				.classTypeIndex(4)
				.separator(",").isSLP(true)
				.hasHeader(true).build();
		
		NeuralNetwork neuralNetwork=new SLP(1, 2, .1, 1, 100);
		ReadSamples samples=ReadSamples.readIRISData(prop, .8);
		/*neuralNetwork.run(samples.getLearnSamples(), samples.getTestSamples());
		neuralNetwork = new LeastMeanSquare(1, 2, .2, 1, .1);
		neuralNetwork.run(samples.getLearnSamples(), samples.getTestSamples());	
		*/samples=ReadSamples.ReadSampleImage("src/main/resources/data", .8);
		
		int[] nNeuralInHidenLayers = { 4 };
		int nHiddenLayers=1;
		int nFeatures=900;
		ActivationType activationType=ActivationType.Sigmoid;
		double stopMse=.001;
		double learningRate=.3;
		double slope=1;
		int nClasses=samples.getNClasses();
		double bias=1;
		//int nPCA=50;
		/*neuralNetwork=new BackPropagation(nClasses, nFeatures, nHiddenLayers, nNeuralInHidenLayers, learningRate, activationType, bias, stopMse, slope);
        neuralNetwork.run(samples.getLearnSamples(), samples.getTestSamples());
        */
        GHA gha=new GHA(50, nFeatures, .001);
        gha.trainPCAUseTestAndLearnSamples(samples.getLearnSamples(), samples.getTestSamples(), 50);
        
        neuralNetwork=new BackPropagation(samples.getNClasses(), 50, nHiddenLayers, nNeuralInHidenLayers, learningRate, activationType, bias, stopMse, slope);
        neuralNetwork.run(gha.regenerateSamples(samples.getLearnSamples()), gha.regenerateSamples(samples.getTestSamples()));
        
        
		
	}

}
