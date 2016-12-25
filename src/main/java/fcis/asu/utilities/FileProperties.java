package fcis.asu.utilities;

import java.util.List;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.Singular;

/**
 * @author Eslam Ali
 * this class to prepare configuration of
 * file path,separator between column,is has Head
 * target classes ,target features columns
 * and classType index in the file
 */
@Builder
@Getter
public class FileProperties {
	
	@NonNull
	private String filePath;
	@NonNull
	private String separator=",";
	private boolean hasHeader = false;
	
	/**
	 *if you read for single layer perceptron this will make the desire 1 or -1 for the each class   
	 */
	private boolean isSLP = true;
	private int classTypeIndex;
	
	/**
	 * targetClasses = null if train and test all classes
	 */
	@Singular
	private List<String> targetClasses = null;
	
	/**
	 * targetFeatures default null if you will use all features
	 */
	@Singular
	private List<Integer> targetFeatures = null;
	
	public int getFeatureIndex(int targetFeaturesIndex){
		if(targetFeatures == null)
			throw new IllegalArgumentException("target features is null");
		return targetFeatures.get(targetFeaturesIndex);
	}
	
}
