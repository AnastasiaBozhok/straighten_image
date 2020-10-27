
/**
 * Inspired by https://github.com/gaborvecsei/Straighten-Image
 * Straighten rotated images.
 * 
 * Created by Anastasia 27/10/2020
 */

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

	private static String outputFolder = "images/out/";

	private static int close_kernel_size = 10;
	private static int dilate_lines_kernel_size = 4;

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
		Mat image = Imgcodecs.imread("images/1.jpg");

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
				new Size(dilate_lines_kernel_size, dilate_lines_kernel_size));
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
				double min_line_length_to_consider = 0.01 * Math.min(binaryImage.width(), binaryImage.height());

				if (distance > min_line_length_to_consider) {
					// Draw line on the "debug" image for visualization
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

		angle = calculateAverageRotationUpToPi(angles, distances);
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

	private static double calculateAverageRotationUpToPi(ArrayList<Double> angles, ArrayList<Double> distances) {

		Double[] dist_arr = toArray(distances);

		int[] sortedIndices = IntStream.range(0, dist_arr.length).boxed()
				.sorted((i, j) -> dist_arr[i].compareTo(dist_arr[j])).mapToInt(ele -> ele).toArray();

		// Do not mind 90� rotation
		if (!angles.isEmpty()) {
			for (int i = 0; i < angles.size(); i++) {
				angles.set(i, angles.get(i) % 90);
			}
		}

		//
		int max_angles_number_to_consider = 10;
		int angles_n = Math.min(sortedIndices.length, max_angles_number_to_consider);
		ArrayList<Double> angles_to_consider = new ArrayList<>(angles_n);
		for (int i = 0; i < angles_n - 1; i++) {
			angles_to_consider.add(angles.get(i));
		}

		// Double mean = calcualteAverage(angles_to_consider);
		// TODO (Anastasia): get rid of outliers
		Double median = findMedian(toArray(angles));

		Double angleToReturn = median;
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

	private Double calcualteAverage(ArrayList<Double> angles) {
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

	// Function for calculating median
	public static double findMedian(Double[] a) {
		int n = a.length;

		// First we sort the array
		Arrays.sort(a);

		// check for even case
		if (n % 2 != 0)
			return (double) a[n / 2];

		return (double) (a[(n - 1) / 2] + a[n / 2]) / 2.0;
	}

}