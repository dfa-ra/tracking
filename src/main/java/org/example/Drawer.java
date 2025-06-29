package org.example;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC2;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Drawer {
    private final List<List<Point>> objs = new ArrayList<>();

    public Drawer() {}

    public void addObject(List<Point> points) {
        List<Point> obj = new ArrayList<>(points);
        objs.add(obj);
    }

    public void draw(Mat image) {
        for (List<Point> obj : objs) {
            Mat pointsMat = new Mat(obj.size(), 1, CV_32SC2);

            IntBuffer buffer = pointsMat.createBuffer();
            for (Point p : obj) {
                buffer.put(p.x());
                buffer.put(p.y());
            }
            buffer.rewind();

            MatVector contours = new MatVector(1);
            contours.put(0, pointsMat);

            polylines(
                    image,
                    contours,
                    true,
                    new Scalar(0, 255, 0, 0),
                    2,
                    LINE_AA,
                    0
            );
        }
        objs.clear();
    }
}
