import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

/**
 * 
 * Vision Pipeline for 2020 season
 *
 */
public class SimpleRetroPipeline {
	
	private static final double METERS_PER_INCH = 0.0254;

    private static final Scalar BLUE = new Scalar(255, 0, 0), GREEN = new Scalar(0, 255, 0),
            RED = new Scalar(0, 0, 255), YELLOW = new Scalar(0, 255, 255), ORANGE = new Scalar(0, 165, 255),
            MAGENTA = new Scalar(255, 0, 255);
    
    private static final double FOCAL_LENGTH = 740; // In pixels, needs tuning if res is changed
    private static final double HFOV = 61;
    private static final double VFOV = 34.3;

    private Mat dst;
    private Mat bitmask;

    private MatOfPoint target;
    private Point[] vertices;
    private Point center;
    
    private Mat intrinsics;
    
    private Mat rvec;
    private Mat tvec;

    public void process(Mat src) {

        if (src.empty()) {
            return;
        }
        
        dst = new Mat();
        src.copyTo(dst);
        
        mask();

        getTarget();
        
        if(target == null) {
        	return;
        }

        getVertices();
        
        if(vertices.length != 4) {
        	return;
        }
        
        getCenter();
        
        solvePnP();
        
        reproject();
    }
    
    private void mask() {
        // Filter in bright objects
        bitmask = new Mat();
        Core.inRange(dst, new Scalar(0, 50, 0), new Scalar(200, 255, 200), bitmask);
    }

    private void getTarget() {
        // Find all external contours
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        try {
            Imgproc.findContours(bitmask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } catch (CvException e) {
            // Sometimes the Mat format gets messed up when switching cameras
            System.out.println(e.getMessage());
        }
        
        // Save the largest
        double largestPerimeter = 0;

        for (MatOfPoint contour : contours) {
            Imgproc.drawContours(dst, Arrays.asList(contour), -1, RED);
        	
          	double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
          	
          	if(perimeter > largestPerimeter) {
          		target = contour;
          		largestPerimeter = perimeter;
          	}
        }
    }
    
    private void getVertices() {    	
        // Get Convex Hull
        MatOfInt indexes = new MatOfInt();
        Imgproc.convexHull(target, indexes);
        MatOfPoint2f hull = extractPoints(target, indexes);
                
        // Get Approximation
      	double perimeter = Imgproc.arcLength(hull, true);
        MatOfPoint2f approx = new MatOfPoint2f();
        Imgproc.approxPolyDP(hull, approx, perimeter * 0.01, true);
        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(approx.toArray())), -1, ORANGE);

        // Get sub-pixel locations of the vertices
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 40, 0.001);
        Imgproc.cornerSubPix(bitmask, approx, new Size(5, 5), new Size(-1, -1), criteria);
        
        // Sort vertices
    	vertices = approx.toArray();
    	Arrays.sort(vertices, new Comparator<Point>() {
			@Override
			public int compare(Point p1, Point p2) {
				return (int)(p1.x - p2.x);
			}
    	});
    	
    	
    	for(Point p : vertices) {
    		Imgproc.drawMarker(dst, p, YELLOW);
    	}
    }
    
    private void getCenter() {
    	double x = (vertices[0].x + vertices[3].x) / 2;
    	double y = (vertices[0].y + vertices[3].y) / 2;
    	
    	center = new Point(x, y);
		Imgproc.drawMarker(dst, center, GREEN);
    }
    
    @SuppressWarnings("unused")
    private void solvePnP() {
        // All camera intrinsics are in pixel values
        double principalOffsetX = dst.width() / 2;
        double principalOffsetY = dst.height() / 2;
        intrinsics = Mat.zeros(3, 3, CvType.CV_64FC1);
        intrinsics.put(0, 0, FOCAL_LENGTH);
        intrinsics.put(0, 2, principalOffsetX);
        intrinsics.put(1, 1, FOCAL_LENGTH);
        intrinsics.put(1, 2, principalOffsetY);
        intrinsics.put(2, 2, 1);

        // 3D axes is same as 2D image axes, right is positive x, down is positive y,
        // forward is positive z (a clockwise axes system), values are in inches
        Point3[] worldPts = new Point3[4];
        worldPts[0] = new Point3(-19.625 * METERS_PER_INCH, 0, 0);
        worldPts[1] = new Point3(-9.8125 * METERS_PER_INCH, 17 * METERS_PER_INCH, 0);
        worldPts[2] = new Point3(9.8125 * METERS_PER_INCH, 17 * METERS_PER_INCH, 0);
        worldPts[3] = new Point3(19.625 * METERS_PER_INCH, 0, 0);

        rvec = new Mat();
        tvec = new Mat();
        Calib3d.solvePnP(new MatOfPoint3f(worldPts), new MatOfPoint2f(vertices), intrinsics, new MatOfDouble(), rvec, tvec);
    }
    
    @SuppressWarnings("unused")
    private void reproject() {
        // Set up and draw 3D box with the corners on the outside of the target
        Point3[] worldPts1 = new Point3[4];
        worldPts1[0] = new Point3(-19.625 * METERS_PER_INCH, 17 * METERS_PER_INCH, 0);
        worldPts1[1] = new Point3(-19.625 * METERS_PER_INCH, -17 * METERS_PER_INCH, 0);
        worldPts1[2] = new Point3(19.625 * METERS_PER_INCH, -17 * METERS_PER_INCH, 0);
        worldPts1[3] = new Point3(19.625 * METERS_PER_INCH, 17 * METERS_PER_INCH, 0);
        MatOfPoint2f reprojPts1 = new MatOfPoint2f();
        Calib3d.projectPoints(new MatOfPoint3f(worldPts1), rvec, tvec, intrinsics, new MatOfDouble(), reprojPts1);

        Point3[] worldPts2 = new Point3[4];
        worldPts2[0] = new Point3(-19.625 * METERS_PER_INCH, 17 * METERS_PER_INCH, -12 * METERS_PER_INCH);
        worldPts2[1] = new Point3(-19.625 * METERS_PER_INCH, -17 * METERS_PER_INCH, -12 * METERS_PER_INCH);
        worldPts2[2] = new Point3(19.625 * METERS_PER_INCH, -17 * METERS_PER_INCH, -12 * METERS_PER_INCH);
        worldPts2[3] = new Point3(19.625 * METERS_PER_INCH, 17 * METERS_PER_INCH, -12 * METERS_PER_INCH);
        MatOfPoint2f reprojPts2 = new MatOfPoint2f();
        Calib3d.projectPoints(new MatOfPoint3f(worldPts2), rvec, tvec, intrinsics, new MatOfDouble(), reprojPts2);

        drawBox(reprojPts1, reprojPts2);
    }
    
    private void drawBox(MatOfPoint2f imagePoints, MatOfPoint2f shiftedImagePoints) {
        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(imagePoints.toArray())), -1, GREEN, 2);

        for (int i = 0; i < imagePoints.rows(); i++) {
            Imgproc.line(dst, new Point(imagePoints.get(i, 0)), new Point(shiftedImagePoints.get(i, 0)), BLUE, 2);
        }

        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(shiftedImagePoints.toArray())), -1, MAGENTA, 2);
    }
    
    private MatOfPoint2f extractPoints(MatOfPoint contour, MatOfInt indexes) {
        int[] arrIndex = indexes.toArray();
        Point[] arrContour = contour.toArray();
        Point[] arrPoints = new Point[arrIndex.length];

        for (int i = 0; i < arrIndex.length; i++) {
            arrPoints[i] = arrContour[arrIndex[i]];
        }

        MatOfPoint2f hull = new MatOfPoint2f(); 
        hull.fromArray(arrPoints);
        return hull; 
    }

    public double getTheta() {
    	if(center == null) {
    		return 0;
    	}
    	
    	return (0.5 - center.x / dst.width()) * HFOV;
    }
    
    public double getPhi() {
    	if(center == null) {
    		return 0;
    	}
    	
    	return (0.5 - center.y / dst.width()) * VFOV;
    }
    
    public String getRVec() {
        String s = "<";
        if (rvec != null) {
            for (int i = 0; i < rvec.rows(); i++) {
                if (i != 0) {
                    s += ", ";
                }
                s += rvec.get(i, 0)[0];
            }
        }
        s += ">";

        return s;
    }

    public String getTVec() {
        String s = "<";
        if (tvec != null) {
            for (int i = 0; i < tvec.rows(); i++) {
                if (i != 0) {
                    s += ", ";
                }
                s += tvec.get(i, 0)[0];
            }
        }
        s += ">";

        return s;
    }

    public Mat getDst() {
        return dst;
    }

}