package org.example;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.example.faceWorker.FaceDetection;
import org.example.utils.CameraUtils;


public class Tracking {
    private final CanvasFrame canvas;
    private final CameraUtils cameraTracking;
    private final FaceDetection faceDetection;
    private final OpenCVFrameConverter.ToMat matConverter;

    public Tracking(String name) throws Exception {

        canvas = new CanvasFrame(name);
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        matConverter = new OpenCVFrameConverter.ToMat();
        faceDetection = new FaceDetection();
        cameraTracking = new CameraUtils(0);
    }

    public void run() throws Exception {
        while (true) {
            Frame frame = cameraTracking.grab();
            if (frame == null) {
                onStop();
                break;
            }
            Mat mat = matConverter.convert(frame);

            if (mat != null) {
                faceDetection.detected(mat, new Drawer());
                canvas.showImage(matConverter.convert(mat));
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
