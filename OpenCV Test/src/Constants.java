import org.opencv.core.MatOfDouble;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;

public class Constants {
    public static final Scalar BLUE = new Scalar(255, 0, 0);
    public static final Scalar GREEN = new Scalar(0, 255, 0);
    public static final Scalar RED = new Scalar(0, 0, 255);
    public static final Scalar YELLOW = new Scalar(0, 255, 255);
    public static final Scalar ORANGE = new Scalar(0, 165, 255);
    public static final Scalar MAGENTA = new Scalar(255, 0, 255);

    public static final int STREAM_RES_SCALE = 1;
    public static final long STREAM_FRAME_TIME_MS = 100;

    public static final double METERS_PER_INCH = 0.0254;

    public static final double INNER_GOAL_OFFSET = 29.25;

    // Our target for practice was made incorrectly
    private static final Point3[] PRACTICE_MODEL_PTS = {
        new Point3(-20.625, 0, 0 - INNER_GOAL_OFFSET),
        new Point3(-11.0, 16.875, 0 - INNER_GOAL_OFFSET),
        new Point3(10.875, 17, 0 - INNER_GOAL_OFFSET),
        new Point3(21.75, 0, 0 - INNER_GOAL_OFFSET)};

    public static final Point3[] MODEL_PTS = PRACTICE_MODEL_PTS;

    // public static final Point3[] MODEL_PTS = {
    //     new Point3(-19.625, 0, 0 - INNER_GOAL_OFFSET),
    //     new Point3(-9.8125, 17, 0 - INNER_GOAL_OFFSET),
    //     new Point3(9.8125, 17, 0 - INNER_GOAL_OFFSET),
    //     new Point3(19.625, 0, 0 - INNER_GOAL_OFFSET)};

    public static final Point3[] REPROJECT_PTS_1 = {
        new Point3(-19.625, 17, 0 - INNER_GOAL_OFFSET),
        new Point3(-19.625, -17, 0 - INNER_GOAL_OFFSET),
        new Point3(19.625, -17, 0 - INNER_GOAL_OFFSET),
        new Point3(19.625, 17, 0 - INNER_GOAL_OFFSET)};

    public static final Point3[] REPROJECT_PTS_2 = {
        new Point3(-19.625, 17, -12 - INNER_GOAL_OFFSET),
        new Point3(-19.625, -17, -12 - INNER_GOAL_OFFSET),
        new Point3(19.625, -17, -12 - INNER_GOAL_OFFSET),
        new Point3(19.625, 17, -12 - INNER_GOAL_OFFSET)};


    // Lifecam-3000 info
    public static final int WIDTH = 640;
    public static final int HEIGHT = 360;

    // public static final double FOCAL_LENGTH = 550; // In pixels for 640 by 480, needs tuning if res is changed
    public static final double HFOV = 61;
    public static final double VFOV = 34.3;

    // Calculated values, RMS = 0.15
    public static final double[][] INTRINSICS_640_BY_360 = {
        {567.50828703, 0, 317.53114845},
        {0, 566.65091873, 181.89179922},
        {0, 0, 1}};

    // public static final MatOfDouble DISTORTION_COEFFS =
    //     new MatOfDouble(new double[]{0, 0, 0, 0, 0});

    public static final MatOfDouble DISTORTION_COEFFS_640_BY_360 =
        new MatOfDouble(new double[]{1.53816821e-01, -1.35258778e+00, 3.43940893e-05, -2.64722740e-03, 2.68753172e+00});
}