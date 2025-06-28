package org.example;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.CvScalar;
import org.bytedeco.opencv.opencv_core.IplImage;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class Tracking {
    private final CanvasFrame canvas;
    private final CameraTracking cameraTracking;
    private final OpenCVFrameConverter.ToIplImage converter;

    public Tracking(String name) throws FrameGrabber.Exception {

        canvas = new CanvasFrame(name);
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        converter = new OpenCVFrameConverter.ToIplImage();
        cameraTracking = new CameraTracking(0);

    }

    public void run() throws Exception {
        while (true) {
            Frame frame = cameraTracking.grab();
            if (frame == null) {
                onStop();
                break;
            }
            IplImage img = converter.convert(frame);
            if (img != null) {
                cvCircle(img, cvPoint(320, 240), 50, CvScalar.RED, 2, CV_AA, 0);
                canvas.showImage(converter.convert(img));
            }
            Thread.sleep(50);
        }
    }

    private void onStop() throws Exception {
        canvas.dispose();
        cameraTracking.stop();
    }

    public static void main(String[] args) throws Exception {
        Tracking videoFrame = new Tracking("Video");
        videoFrame.run();
    }
}
