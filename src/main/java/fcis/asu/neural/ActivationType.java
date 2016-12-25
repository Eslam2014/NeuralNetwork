package fcis.asu.neural;

public enum ActivationType {
	Signum("Signum"),Sigmoid("Logistic Sigmoid"),Tangent("Hyperbolic tangent"),Linear("Linear");
	private String name;
	private ActivationType(String name) {
		this.name=name;
	}
	
	@Override
	public String toString() {
		return this.name;
	}
	

}
