import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class ArrayListUtils {

	public static void sortArraysByFirstArray(ArrayList<Double> arrayToSortBy, ArrayList<Double> array2,
			ArrayList<Double> arrayOneSortedByFirst, ArrayList<Double> arrayTwoSortedByFirst) {
		// Output arrays has to be initialised !

		// Keep indices to also sort the 2nd array
		Double[] anglesArray = toArray(arrayToSortBy);
		int[] sortedIndicesAngles = IntStream.range(0, anglesArray.length).boxed()
				.sorted((i, j) -> anglesArray[i].compareTo(anglesArray[j])).mapToInt(ele -> ele).toArray();
		// Sort
		for (int i = 0; i < arrayToSortBy.size(); i++) {
			arrayOneSortedByFirst.add(arrayToSortBy.get(sortedIndicesAngles[i]));
			arrayTwoSortedByFirst.add(array2.get(sortedIndicesAngles[i]));
		}
	}

	public static Double[] toArray(ArrayList<Double> arrList) {
		Double[] dist_arr_for_copy = new Double[arrList.size()];
		dist_arr_for_copy = arrList.toArray(dist_arr_for_copy);
		Double[] dist_arr = dist_arr_for_copy.clone();
		return dist_arr;
	}

	public static Double calcualteAverage(ArrayList<Double> array) {
		Double sum = 0d;
		if (!array.isEmpty()) {
			for (Double angle : array) {
				sum += angle;
			}
			return sum / array.size();
		}
		return sum;
	}

	// https://stackoverflow.com/questions/18805178/how-to-detect-outliers-in-an-arraylist
	public static List<Integer> getOutliersIndicesOfSortedList(List<Double> input) {
		List<Integer> outputIndices = new ArrayList<Integer>();
		List<Double> data1 = new ArrayList<Double>();
		List<Double> data2 = new ArrayList<Double>();
		if (input.size() % 2 == 0) {
			data1 = input.subList(0, input.size() / 2);
			data2 = input.subList(input.size() / 2, input.size());
		} else {
			data1 = input.subList(0, input.size() / 2);
			data2 = input.subList(input.size() / 2 + 1, input.size());
		}
		double q1 = getMedianOfSortedList(data1);
		double q3 = getMedianOfSortedList(data2);
		double iqr = q3 - q1;
		double lowerFence = q1 - 1.5 * iqr;
		double upperFence = q3 + 1.5 * iqr;
		for (int i = 0; i < input.size(); i++) {
			if (input.get(i) < lowerFence || input.get(i) > upperFence)
				outputIndices.add(i);
		}
		return outputIndices;
	}

	public static double getMedianOfSortedList(List<Double> data) {
		if (data.size() % 2 == 0)
			return (data.get(data.size() / 2) + data.get(data.size() / 2 - 1)) / 2;
		else
			return data.get(data.size() / 2);
	}

	public static double getStandartDeviation(ArrayList<Double> array) {
		double mean = 0.0;
		double var_sum = 0.0;
		double numi = 0.0;

		mean = calcualteAverage(array);

		for (Double el : array) {
			numi = Math.pow((el - mean), 2);
			var_sum += numi;
		}

		return Math.sqrt(var_sum / array.size());
	}
}