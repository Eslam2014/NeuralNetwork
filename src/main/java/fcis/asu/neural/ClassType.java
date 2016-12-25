package fcis.asu.neural;

import lombok.Getter;

@Getter
public class ClassType {
	private String name;
	private int[] desire;

	public ClassType(String name, int[] desire) {
		this.name = name;
		this.desire = desire;
	}
	
	@Override
	public String toString() {
		return name;
	}
	
	

}
