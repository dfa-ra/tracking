package org.example.faceWorker;

import org.bytedeco.javacpp.indexer.FloatIndexer;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.example.Drawer;
import org.example.utils.ResourceUtils;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.opencv.imgproc.Imgproc.LINE_8;


public class FaceDetection {

    private final File model = ResourceUtils.extractToTempFile("models/res10_300x300_ssd_iter_140000.caffemodel");
    private final File proto = ResourceUtils.extractToTempFile("proto/deploy.prototxt");

    private final Net net = readNetFromCaffe(proto.getAbsolutePath(), model.getAbsolutePath());

    public FaceDetection() throws IOException {

    }

    public void detected(Mat image, Drawer drawer) {
        Mat inputBlob = blobFromImage(image, 1.0, new Size(300, 300),
                new Scalar(104.0, 177.0, 123.0, 0.0), false, false, CV_32F);
        net.setInput(inputBlob);

        Mat detections = net.forward();
        FloatIndexer indexer = detections.createIndexer();
        int cols = image.cols();
        int rows = image.rows();
        System.out.println(cols + " " + rows);
        for (int i = 0; i < detections.size(2); i++) {

            float confidence = indexer.get(0, 0, i, 2);

            if (confidence > 0.5) {


                int x1 = Math.round(indexer.get(0, 0, i, 3) * cols);
                int y1 = Math.round(indexer.get(0, 0, i, 4) * rows);
                int x2 = Math.round(indexer.get(0, 0, i, 5) * cols);
                int y2 = Math.round(indexer.get(0, 0, i, 6) * rows);
                System.out.println(x1  + ":" + y1 + " | " + x2 + ":" + y2);
                List<Point> points = new ArrayList<>();
                points.add(new Point(x1, y1));
                points.add(new Point(x1, y2));
                points.add(new Point(x2, y2));
                points.add(new Point(x2, y1));

                drawer.addObject(points);
//                rectangle(image, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0, 0), 2, LINE_8, 0);
            }

        }
        drawer.draw(image);
    }
}
