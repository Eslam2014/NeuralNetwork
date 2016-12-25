package fcis.asu.utilities;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;

import fcis.asu.neural.ClassType;
import fcis.asu.neural.Sample;
import lombok.Cleanup;
import lombok.Getter;
import lombok.NonNull;

public class ReadSamples {
	@Getter private Map<double[], ClassType> testSamples;
	@Getter private Map<double[], ClassType> learnSamples;
	@Getter private int nClasses;
	@Getter public static ClassType[] classTypes;
	final static Logger logger = Logger.getLogger(ReadSamples.class);

	private ReadSamples(String rootFolderPath, double percentageSampleLearn) {
		testSamples = new HashMap<double[], ClassType>();
		learnSamples = new HashMap<double[], ClassType>();
		generateLearnAndTestSamples(rootFolderPath, percentageSampleLearn);
	}

	private ReadSamples(FileProperties properties,double percentageSampleLearn) {
		testSamples = new HashMap<double[], ClassType>();
		learnSamples = new HashMap<double[], ClassType>();
		if (properties.isSLP()) {
			classTypes = new ClassType[2];
			classTypes[0] = new ClassType(properties.getTargetClasses().get(0), new int[] { 1 });
			classTypes[1] = new ClassType(properties.getTargetClasses().get(1), new int[] { -1 });
		}
		readIRIS(properties, percentageSampleLearn);
	}

	public static ReadSamples ReadSampleImage(@NonNull String rootFolderPath, double percentageSampleLearn) {
		return new ReadSamples(rootFolderPath, percentageSampleLearn);
	}

	public static ReadSamples readIRISData(@NonNull FileProperties properties,double percentageSampleLearn) {
		return new ReadSamples(properties, percentageSampleLearn);
	}

	

	/**
	 * this function real every subfolder of the root folder and generate
	 * ClassType object for it . then loop in each class folder to set data in
	 * map<String : image path ,ClassType : class of the image> learn and test
	 * 
	 * @param rootFolderPath
	 *            this folder should contain folder for each class
	 * @param percentageSampleLearn
	 *            percentage of samples you to use for learn the rest will use
	 *            for test
	 */
	private void generateLearnAndTestSamples(String rootFolderPath, double percentageSampleLearn) {
		File file = new File(rootFolderPath);
		String[] classesNames = file.list(new FilenameFilter() {
			public boolean accept(File current, String name) {
				return new File(current, name).isDirectory();
			}
		});

		this.nClasses = classesNames.length;
		classTypes = new ClassType[nClasses];
		for (int i = 0; i < nClasses; i++) {
			int[] desire = new int[nClasses];
			desire[i] = 1;
			classTypes[i] = new ClassType(classesNames[i], desire);

			String classPath = rootFolderPath + "/" + classesNames[i];
			File Classdir = new File(classPath);
			File[] samples = Classdir.listFiles(new FilenameFilter() {
				public boolean accept(File dir, String name) {
					return name.endsWith(".jpg");
				}
			});
			int nSamples = samples.length;
			int nLeanrnSample = (int) Math.round((double) nSamples * percentageSampleLearn);

			for (int j = 0; j < nLeanrnSample; j++) {
				learnSamples.put(getImgData(samples[j].getPath()), classTypes[i]);
			}

			for (int j = nLeanrnSample; j < nSamples; j++) {
				testSamples.put(getImgData(samples[j].getPath()), classTypes[i]);
			}
		}
	}

	public static double[] getImgData(@NonNull String imgPath) {
		int height, width, rgb, red, green, blue, count;
		double imgData[] = null;
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(imgPath));
			width = img.getWidth();
			height = img.getHeight();
			imgData = new double[width * height];
			count = 0;
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					rgb = img.getRGB(w, h);
					red = (rgb >> 16) & 0x000000FF;
					green = (rgb >> 8) & 0x000000FF;
					blue = (rgb) & 0x000000FF;
					double sum = (double) (red + green + blue) / 3;
					imgData[count] = sum;
					count++;
				}
			}
		} catch (IOException e) {
			System.out.println(e);
		}
		// System.out.println(imgPath);
		return Utilities.normalize(imgData);

	}
		
	
	private void readIRIS(FileProperties prop,double percentageSampleLearn) {
		try {
			@Cleanup
			BufferedReader in = new BufferedReader(new FileReader(prop.getFilePath()));
			
			String row;
			int noisyRows = 0;
			int totalRows = 0;		
			List<Integer> TargetFeatures=prop.getTargetFeatures();
						
			if (prop.isHasHeader()) {
				in.readLine();
			}
			int nFeatures = TargetFeatures.size();
			List<Sample> samples=new ArrayList<Sample>();
			double[] features;
			while ((row = in.readLine()) != null) {
				String[] columns = row.split(prop.getSeparator());
            
				try {
					String className = columns[prop.getClassTypeIndex()];
					ClassType type = getClassType(className);
					if(prop.isSLP()){
						if(type != null){
							 features = new double[nFeatures];
							 for (int i = 0; i < nFeatures; i++) {
									features[i] = Double.parseDouble(columns[prop.getTargetFeatures().get(i)])/10;
							}
							 samples.add(new Sample(type, features));
						}
					}
					else {
						throw new UnsupportedOperationException("sorry not handeled yet");
					}
			
				} catch (Exception e) {
					logger.error(e);
					noisyRows++;

				}
				totalRows++;
			}
			logger.info(
					"Total rows: " + totalRows + " mapped to sample:" + samples.size() + " noisy rows:" + noisyRows);
		
			
			generateTestAndLearnSamplesIRIS(samples, percentageSampleLearn);

		} catch (IOException err) {
			logger.error("file read error :" + err.getMessage());
			System.exit(0);
			
		}

	}

	private void generateTestAndLearnSamplesIRIS(List<Sample> samples, double percentageSampleLearn) {
		List<Sample> listLearnSamples = new ArrayList<Sample>();
		List<Sample> listTestSamples = new ArrayList<Sample>();
		int nClassSamples = samples.size() / 2;
		
		int nLearnSamples = (int) Math.round(nClassSamples * percentageSampleLearn);
		listLearnSamples.addAll(samples.subList(0, nLearnSamples - 1));
		listLearnSamples.addAll(samples.subList(nClassSamples - 1, nClassSamples + nLearnSamples - 2));
		for(Sample sample:listLearnSamples){
			learnSamples.put( sample.getFeatures(),sample.getClassType());
		}
		
		listTestSamples.addAll(samples.subList(nLearnSamples, nClassSamples - 1));
		listTestSamples.addAll(samples.subList(nClassSamples + nLearnSamples - 1, samples.size() - 1));
		for(Sample sample:listTestSamples){
			testSamples.put(sample.getFeatures(),sample.getClassType());
		}
	}
	
	private static ClassType getClassType(String className) {
		for (ClassType type : classTypes) {
			if (type.toString().equals(className))
				return type;
		}
		return null;
	}

}
