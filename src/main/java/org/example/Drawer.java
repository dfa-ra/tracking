package org.example;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Scalar;

import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Drawer {

    public Drawer() {}

    public void draw(Mat image, List<Mat> objsMat) {
        for (Mat obj : objsMat) {

            MatVector contours = new MatVector(1);
            contours.put(0, obj);

            polylines(
                    image,
                    contours,
                    false,
                    new Scalar(0, 255, 0, 0),
                    2,
                    LINE_AA,
                    0
            );
        }
    }
}
