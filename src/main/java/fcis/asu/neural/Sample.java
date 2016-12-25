package fcis.asu.neural;

import lombok.Getter;

@Getter
public class Sample {
	private ClassType classType;
	private double[] features;
	
	public Sample(ClassType classType,double[] features) {
		this.classType=classType;
		this.features=features;
	}

}
