import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

public class Main {

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Replace this String with the path to your vision image
		Mat matrix = Imgcodecs.imread("C:\\Users\\Junqi\\Downloads\\2020 Vision Images\\BlueGoal-330in-ProtectedZone.jpg");

		
		SimpleRetroPipeline rp = new SimpleRetroPipeline();

		
		rp.process(matrix);

		HighGui.imshow("src", matrix);
		HighGui.imshow("dst", rp.getDst());
		
		HighGui.waitKey();
	}

	
}
