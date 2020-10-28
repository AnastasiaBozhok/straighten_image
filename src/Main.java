
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

import org.opencv.core.Core;
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

	private static final String inputFolder = "images/in/";
	private static final String outputFolder = "images/out/";
	private static final String debugFolder = "images/debug/";

	// 0 -- use all detected lines, 1 -- only those of the image size (probably
	// none)
	private static final double min_line_length_to_consider_percentage = 0.02;
	private static final int min_line_length_to_consider_pixels = 10;
	private static final double close_kernel_size_as_percentage = 0.01;
	private static final int min_close_kernel_size_pixels = 10;
	private static final int dilate_lines_kernel_size_pixels = 4;
	private static final int max_angles_number_to_consider = 20;

	public static void main(String[] args) {

		// For OpenCV (this is compulsory)
		nu.pattern.OpenCV.loadLocally();
		// System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);

		System.out.println("Started");

		// Creating the output directories
		new File(outputFolder).mkdir();
		new File(debugFolder).mkdir();

		// Get the images
		File folder = new File(inputFolder);
		File[] listOfFiles = folder.listFiles();
		for (File file : listOfFiles) {
			if (file.isFile()) {
				String fileName = file.getName();

				// read the image
				Mat image = Imgcodecs.imread(inputFolder + fileName);
				if (image.width() == 0) {
					System.out.println("Problem reading the image from '" + fileName + "'. Exit.");
				} else {

					System.out.println("Treating the image " + fileName + "...");

					// Straight it out
					Mat straightImage = straightenImage(image);

					// Write the results
					Imgcodecs.imwrite(outputFolder + fileName, straightImage);
				}
			}
		}

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
		Imgcodecs.imwrite(debugFolder + "threshold.jpg", binary);

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
		Imgcodecs.imwrite(debugFolder + "afterClose.jpg", binary);

		// Edge detection
		Imgproc.Canny(binary, binary, 50, 200, 3, false);
		Imgcodecs.imwrite(debugFolder + "afterCanny.jpg", binary);

		// Dilate lines width for easier detection
		kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
				new Size(dilate_lines_kernel_size_pixels, dilate_lines_kernel_size_pixels));
		Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_DILATE, kernel);

		Imgcodecs.imwrite(debugFolder + "processedImage.jpg", binary);
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
					if (angle < 0)
						angle = angle + 180;

					angles.add(angle);
					distances.add(distance);
				}
			}
		}

		Imgcodecs.imwrite(debugFolder + "detectedLines.jpg", debugImage);

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

		// First we sort the angles
		ArrayList<Double> anglesSortedByAngles = new ArrayList<>();
		ArrayList<Double> distancesSortedByAngles = new ArrayList<>();
		ArrayUtils.sortArraysByFirstArray(angles, distances, anglesSortedByAngles, distancesSortedByAngles);

		// Remove clearly aberrant angles
		List<Integer> outliersIndices = ArrayUtils.getOutliersIndicesOfSortedList(anglesSortedByAngles);
		for (int i = outliersIndices.size() - 1; i >= 0; i--) {
			anglesSortedByAngles.remove((int) outliersIndices.get(i));
			distancesSortedByAngles.remove((int) outliersIndices.get(i));
		}

		// Sort distances from shortest to longest (to find the longest lines)
		ArrayList<Double> anglesSortedByDistance = new ArrayList<>();
		ArrayList<Double> distancesSortedByDistance = new ArrayList<>();
		ArrayUtils.sortArraysByFirstArray(distancesSortedByAngles, anglesSortedByAngles, distancesSortedByDistance,
				anglesSortedByDistance);
		// From longest to shortest
		Collections.reverse(anglesSortedByDistance);

		// Consider only max_angles_number_to_consider longest lines
		int angles_n = Math.min(anglesSortedByDistance.size(), max_angles_number_to_consider);
		ArrayList<Double> angles_to_consider = new ArrayList<>(angles_n);
		for (int i = 0; i < angles_n; i++) {
			angles_to_consider.add(anglesSortedByDistance.get(i));
		}

		// TODO: Test both options
		Double angleToReturn = ArrayUtils.getMedianOfSortedList(angles_to_consider);
//		Double angleToReturn = calcualteAverage(angles_to_consider);

		if (Math.abs(angleToReturn - 90) < angleToReturn)
			angleToReturn = angleToReturn - 90;

		return angleToReturn;
	}

	// https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
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
//			Imgcodecs.imwrite(outputFolder + "shadow1.jpg", plane);
			
			// Dilate
			Imgproc.morphologyEx(plane, binary, Imgproc.MORPH_DILATE, kernel);
//			Imgcodecs.imwrite(outputFolder + "shadow2.jpg", binary);
			Imgproc.medianBlur(binary, bg_img, 21);
//			Imgcodecs.imwrite(outputFolder + "shadow3.jpg", bg_img);
			Core.absdiff(plane, bg_img, diff_img);
//			Imgcodecs.imwrite(outputFolder + "shadow4.jpg", diff_img);
			Core.absdiff(diff_img, new Scalar(255), diff_img);
//			Imgcodecs.imwrite(outputFolder + "shadow5.jpg", diff_img);
			Core.normalize(diff_img, norm_img, 0, 255, Core.NORM_MINMAX);
//			Imgcodecs.imwrite(outputFolder + "shadow6.jpg", norm_img);

			result_planes.add(diff_img);
			result_norm_planes.add(norm_img);
		}

		Core.merge(result_planes, result);
		Core.merge(result_norm_planes, result);

		Imgcodecs.imwrite(debugFolder + "shadowless.jpg", result);
		return result;
	}
}