package org.example;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.example.face.FaceDetection;
import org.example.face.FaceTracking;
import org.example.hands.HandsDetection;
import org.example.utils.CameraUtils;

import java.time.LocalTime;
import java.util.List;


public class Tracking {
    private final CanvasFrame canvas;
    private final CameraUtils cameraUtils;
    private final FaceDetection faceDetection;
    private final HandsDetection handsDetection;
    private final FaceTracking faceTracking;
    private final Drawer drawer;
    private final OpenCVFrameConverter.ToMat matConverter;

    public Tracking(String name) throws Exception {

        canvas = new CanvasFrame(name);
        drawer = new Drawer();
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        matConverter = new OpenCVFrameConverter.ToMat();

        faceDetection = new FaceDetection();
        faceTracking = new FaceTracking();

        handsDetection = new HandsDetection();

        cameraUtils = new CameraUtils(0);
    }

    public void run() throws Exception {
        LocalTime now = LocalTime.now();
        while (true) {
            Frame frame = cameraUtils.grab();
            if (frame == null) {
                onStop();
                break;
            }
            Mat mat = matConverter.convert(frame);

            if (mat != null) {
                List<Mat> hands = handsDetection.detect(mat);
                if (!hands.isEmpty())
                    drawer.draw(mat, hands);
//                if (!faceTracking.update(mat) || (LocalTime.now().getSecond() - now.getSecond()) >= 1){
//                    now = LocalTime.now();
//                    List<Mat> faces = faceDetection.detect(mat);
//                    if (!faces.isEmpty()) faceTracking.init(mat, faces);
//                }
//
//                List<Mat> faces = faceTracking.getMatObjects();
//                if (!faces.isEmpty())
//                    drawer.draw(mat, faces);
                canvas.showImage(matConverter.convert(mat));
            }
            Thread.sleep(50);
        }
    }

    private void onStop() throws Exception {
        canvas.dispose();
        cameraUtils.stop();
    }

    public static void main(String[] args) throws Exception {
        Tracking videoFrame = new Tracking("Video");
        videoFrame.run();
    }
}
