package fcis.asu.utilities;

import java.util.Random;

import lombok.NonNull;

public class Utilities {

	/**
	 * generate array of random numbers
	 * 
	 * @param nNumbers must be greater than 0
	 * @return random Double arrays between 0 : 1
	 */
	public static double[] generateRandomDoubleArr(int nNumbers) {
		if(nNumbers == 0)
		throw new IllegalArgumentException("nNumber must be greater than 0");
		
		Random rand = new Random();
		double[] randomNumbers = new double[nNumbers];
		for (int i = 0; i < nNumbers; i++) {
			randomNumbers[i] = rand.nextFloat();
		}
		return randomNumbers;
	}

	/**
	 * generate 2D array of random numbers
	 * 
	 * @param x
	 *            numbers in x dimensional
	 * @param y
	 *            numbers in y dimensional
	 * @return random Double 2D array between 0 : 1
	 */
	public static double[][] generateRandomDouble2DArr(int x, int y) {
		if(x == 0 || y== 0)
			throw new IllegalArgumentException("x and y must be greater than 0");
			
		Random rand = new Random();
		double[][] rand2DArr = new double[x][y];
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				rand2DArr[i][j] = rand.nextFloat();
			}

		}
		return rand2DArr;
	}

	/**
	 * The dot product of two vectors a = [a1, a2, ..., an] and b = [b1, b2,...,
	 * bn] is a1*b1 + a2*b2 + a3*b3 +...
	 * 
	 * @param a
	 * @param b
	 * @return dot product or scalar product
	 * 
	 */
	public static double dotProduct(@NonNull double[] a, @NonNull double[] b) {
		if (a.length != b.length)
			throw new IllegalArgumentException("arrays must have equal lengths");
		
		int arrLength = a.length;
		double sum = 0;
		for (int i = 0; i < arrLength; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}
	
	
	/**
	 * Normalize double array data 
	 * @param arr
	 * @return normalize array
	 */
	public static double[] normalize(@NonNull double[] arr) {				
		double sum = 0.0;
		double max = 0.0;
		double tmpNum;
		for (double num : arr) {
			sum += num;
			tmpNum=Math.abs(num);
            max = tmpNum> max ? tmpNum : max;
		}
		
		int lenght = arr.length;
		double mean = sum / lenght;
		double[] newArr = new double[lenght];
		for (int i = 0; i < lenght; i++) {
			double f = (arr[i] - mean) / max;
			newArr[i] = f;
		}

		return newArr;
	}
}



