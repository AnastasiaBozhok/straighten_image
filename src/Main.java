
/**
 * Inspired by https://github.com/gaborvecsei/Straighten-Image
 * Straighten rotated images.
 * 
 * Created by Anastasia 27/10/2020
 */

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.IntStream;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

	private static final String outputFolder = "images/out/";

	// 0 -- use all detected lines, 1 -- only those of the image size (probably
	// none)
	private static final double min_line_length_to_consider_percentage = 0.02;
	private static final int min_line_length_to_consider_pixels = 10;
	private static final double close_kernel_size_as_percentage = 0.01;
	private static final int min_close_kernel_size_pixels = 10;
	private static final int dilate_lines_kernel_size_pixels = 4;

	public static void main(String[] args) {
		// For OpenCV (this is compulsory)
		// nu.pattern.OpenCV.loadShared();
		nu.pattern.OpenCV.loadLocally();
		// System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);

		System.out.println("Started");

		// Creating the output directory
		File file = new File(outputFolder);
		file.mkdir();

		// read our original image
//		String fileName = "images/1.jpg";
//		String fileName = "images/boxesOCR/barcodeExample.png";
//		String fileName = "images/boxesOCR/5.jpg"; // problem
		String fileName = "images/boxesOCR/2.jpg";
		
		Mat image = Imgcodecs.imread(fileName);
		if (image.width() == 0) {
			System.out.println("Problem reading the image from '" + fileName + "'. Exit.");
			return;
		}

		// Straight it out! :)
		Mat straightImage = straightenImage(image);
		Imgcodecs.imwrite(outputFolder + "straightImage.jpg", straightImage);

		System.out.println("Finished");
	}

	// This is the pre-processing part where we create a binary image from our
	// original
	// And after the morphology we can detect the test parts more easily
	private static Mat preProcessForAngleDetection(Mat image) {
		Mat binary = new Mat();

		// Convert the image to gray from RGB
		Mat grayscale = new Mat(image.height(), image.width(), CvType.CV_8UC1);
		Imgproc.cvtColor(image, grayscale, Imgproc.COLOR_RGB2GRAY);

		// Create binary image
		int max_value = 255; // TODO: find grayscale.max()
		Imgproc.threshold(grayscale, binary, max_value / 2, max_value, Imgproc.THRESH_BINARY_INV);
		Imgcodecs.imwrite(outputFolder + "threshold.jpg", binary);

		// "Connect" the letters and words
		// initially only one Mat kernel =
		// Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 1));
		int close_kernel_size = (int) Math
				.round(close_kernel_size_as_percentage * Math.min(image.height(), image.width()));
		close_kernel_size = Math.max(close_kernel_size, min_close_kernel_size_pixels);

		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(close_kernel_size, 1));
		Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
		kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, close_kernel_size));
		Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
		Imgcodecs.imwrite(outputFolder + "afterClose.jpg", binary);

		// Edge detection
		Imgproc.Canny(binary, binary, 50, 200, 3, false);
		Imgcodecs.imwrite(outputFolder + "afterCanny.jpg", binary);

		// Dilate lines width for easier detection
		kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
				new Size(dilate_lines_kernel_size_pixels, dilate_lines_kernel_size_pixels));
		Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_DILATE, kernel);

		Imgcodecs.imwrite(outputFolder + "processedImage.jpg", binary);
		return binary;
	}

	// With this we can detect the rotation angle
	// After this function returns we will know the necessary angle
	private static double detectRotationAngle(Mat binaryImage) {
		// Store line detections here
		Mat lines = new Mat();
		// Detect lines
		Imgproc.HoughLinesP(binaryImage, lines, 1, Math.PI / 180, 100);

		double angle = 0;

		// This is only for debugging and to visualise the process of the straightening
		Mat debugImage = binaryImage.clone();
		Imgproc.cvtColor(debugImage, debugImage, Imgproc.COLOR_GRAY2BGR);

		double min_line_length_to_consider = min_line_length_to_consider_percentage
				* Math.min(binaryImage.width(), binaryImage.height());
		min_line_length_to_consider = Math.max(min_line_length_to_consider, min_line_length_to_consider_pixels);

		ArrayList<Double> angles = new ArrayList<Double>();
		ArrayList<Double> distances = new ArrayList<Double>();
		// Calculate the start and end point and the angle
		for (int x = 0; x < lines.cols(); x++) {
			for (int y = 0; y < lines.rows(); y++) {
				double[] vec = lines.get(y, x);
				double x1 = vec[0];
				double y1 = vec[1];
				double x2 = vec[2];
				double y2 = vec[3];

				Point start = new Point(x1, y1);
				Point end = new Point(x2, y2);

				double distance = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
				if (distance > min_line_length_to_consider) {
					// Draw line on the "debug" image for visualisation
					Imgproc.line(debugImage, start, end, new Scalar(255, 255, 0), 5);

					// Calculate the angle we need
					angle = calculateAngleFromPoints(start, end);
					if (angle < 0) {
						angle = angle + 180;
					}

					angles.add(angle);
					distances.add(distance);
				}
			}
		}

		Imgcodecs.imwrite(outputFolder + "detectedLines.jpg", debugImage);

		angle = estimateRotationAngleUpToPi(angles, distances);
		return angle;
	}

	// From an end point and from a start point we can calculate the angle
	private static double calculateAngleFromPoints(Point start, Point end) {
		double deltaX = end.x - start.x;
		double deltaY = end.y - start.y;
		return Math.atan2(deltaY, deltaX) * (180 / Math.PI);
	}

	// Rotation is done here
	private static Mat rotateImage(Mat image, double angle) {
		// Calculate image center
		Point imgCenter = new Point(image.cols() / 2, image.rows() / 2);
		// Get the rotation matrix
		Mat rotMtx = Imgproc.getRotationMatrix2D(imgCenter, angle, 1.0);
		// Calculate the bounding box for the new image after the rotation (without this
		// it would be cropped)
		Rect bbox = new RotatedRect(imgCenter, image.size(), angle).boundingRect();

		// Rotate the image
		Mat rotatedImage = image.clone();
		Imgproc.warpAffine(image, rotatedImage, rotMtx, bbox.size());

		return rotatedImage;
	}

	// Sums the whole process and returns with the straight image
	private static Mat straightenImage(Mat image) {
		Mat rotatedImage = image.clone();
		Mat imageShadowless = remove_shadow(image.clone());
		Mat processed = preProcessForAngleDetection(imageShadowless);
		double rotationAngle = detectRotationAngle(processed);

		return rotateImage(rotatedImage, rotationAngle);
	}

	private static double estimateRotationAngleUpToPi(ArrayList<Double> angles, ArrayList<Double> distances) {

		// Do not mind 90° rotation
		if (!angles.isEmpty()) {
			for (int i = 0; i < angles.size(); i++) {
				angles.set(i, angles.get(i) % 90);
			}
		}
		
		// First we sort the angles (keep indices to also sort distances)
		Double[] anglesArray = toArray(angles);
		int[] sortedIndicesAngles = IntStream.range(0, anglesArray.length).boxed()
				.sorted((i, j) -> anglesArray[i].compareTo(anglesArray[j])).mapToInt(ele -> ele).toArray();
		ArrayList<Double> anglesSortedByAngles = new ArrayList<>();
		ArrayList<Double> distancesSortedByAngles = new ArrayList<>();		
		for (int i = 0; i < angles.size(); i++) {
			anglesSortedByAngles.add(angles.get(sortedIndicesAngles[i]));
			distancesSortedByAngles.add(distances.get(sortedIndicesAngles[i]));
		}

		// Remove clearly aberrant angles
		List<Integer> outliersIndices = getOutliersIndicesOfSortedList(anglesSortedByAngles);
		for (int i = outliersIndices.size()-1; i >= 0; i--) {
			anglesSortedByAngles.remove((int) outliersIndices.get(i));
			distancesSortedByAngles.remove((int) outliersIndices.get(i));
		}

		// Sort distances (keep indices to also sort angles)
		Double[] distArray = toArray(distancesSortedByAngles);
		int[] sortedIndicesDistances = IntStream.range(0, distArray.length).boxed()
				.sorted((i, j) -> distArray[i].compareTo(distArray[j])).mapToInt(ele -> ele).toArray();
		ArrayList<Double> anglesSortedByDistance = new ArrayList<>();
		ArrayList<Double> distancesSortedByDistance = new ArrayList<>();		
		for (int i = sortedIndicesDistances.length - 1; i >= 0; i--) {
			anglesSortedByDistance.add(anglesSortedByAngles.get(sortedIndicesDistances[i]));
			distancesSortedByDistance.add(distancesSortedByAngles.get(sortedIndicesDistances[i]));
		}

		// Consider only max_angles_number_to_consider longest lines
		int max_angles_number_to_consider = 20;
		int angles_n = Math.min(anglesSortedByDistance.size(), max_angles_number_to_consider);
		ArrayList<Double> angles_to_consider = new ArrayList<>(angles_n);
		for (int i = 0; i < angles_n - 1; i++) {
			angles_to_consider.add(anglesSortedByDistance.get(i));
		}
		
		// TODO (Anastasia): get rid of outliers
//		Double angleToReturn = getMedianOfSortedList(angles);
		Double angleToReturn = getMedianOfSortedList(angles_to_consider);
//		Double angleToReturn = calcualteAverage(angles_to_consider);

		if (Math.abs(angleToReturn - 90) < angleToReturn)
			angleToReturn = angleToReturn - 90;

		return angleToReturn;
	}

	private static Double[] toArray(ArrayList<Double> arrList) {
		Double[] dist_arr_for_copy = new Double[arrList.size()];
		dist_arr_for_copy = arrList.toArray(dist_arr_for_copy);
		Double[] dist_arr = dist_arr_for_copy.clone();
		return dist_arr;
	}

	private static Double calcualteAverage(ArrayList<Double> angles) {
		Double sum = 0d;
		if (!angles.isEmpty()) {
			for (Double angle : angles) {
				sum += angle % 90;
			}
			return sum / angles.size();
		}
		return sum;
	}

	private static Mat remove_shadow(Mat image) {
		List<Mat> rgb_planes = new ArrayList<Mat>();
		Core.split(image, rgb_planes);
		Mat binary = new Mat();
		Mat bg_img = new Mat();
		Mat diff_img = new Mat();
		Mat norm_img = new Mat();
		List<Mat> result_planes = new ArrayList<Mat>();
		List<Mat> result_norm_planes = new ArrayList<Mat>();
		Mat result = new Mat();

		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));

		for (Mat plane : rgb_planes) {

			// Dilate
			Imgproc.morphologyEx(plane, binary, Imgproc.MORPH_DILATE, kernel);
			Imgproc.medianBlur(binary, bg_img, 21);
			Core.absdiff(plane, bg_img, diff_img);
			Core.absdiff(diff_img, new Scalar(255), diff_img);
			Core.normalize(diff_img, norm_img, 0, 255, Core.NORM_MINMAX);

			result_planes.add(diff_img);
			result_norm_planes.add(norm_img);
		}

		Core.merge(result_planes, result);
		Core.merge(result_norm_planes, result);

		Imgcodecs.imwrite(outputFolder + "shadowless.jpg", result);
		return result;
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

	private static double getMedianOfSortedList(List<Double> data) {
		if (data.size() % 2 == 0)
			return (data.get(data.size() / 2) + data.get(data.size() / 2 - 1)) / 2;
		else
			return data.get(data.size() / 2);
	}
}
