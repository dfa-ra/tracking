package org.example.face;

import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_tracking.TrackerKCF;


import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC2;

public class FaceTracking {
    private final List<TrackerKCF> trackers = new ArrayList<>();
    private final List<Rect> objects = new ArrayList<>();

    public FaceTracking() {}

    public void init(Mat image, List<Mat> frames) {
        clear();
        System.out.println("Реинициализация трекера");
        for (Mat frame : frames) {
            IntIndexer indexer = frame.createIndexer();

            int minX = Integer.MAX_VALUE;
            int minY = Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE;
            int maxY = Integer.MIN_VALUE;

            for (int i = 0; i < frame.rows(); i++) {
                int x = indexer.get(i, 0, 0);
                int y = indexer.get(i, 0, 1);
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }

            Rect roi = new Rect(minX, minY, maxX - minX, maxY - minY);

            if (!image.empty() &&
                    roi.width() > 0 && roi.height() > 0 &&
                    roi.x() >= 0 && roi.y() >= 0 &&
                    roi.x() + roi.width() <= image.cols() &&
                    roi.y() + roi.height() <= image.rows()) {

                TrackerKCF tracker = TrackerKCF.create();
                tracker.init(image, roi);

                trackers.add(tracker);
                objects.add(roi);

            } else {
                System.err.println("Invalid ROI or image — skipping tracker init.");
            }

        }
    }

    public boolean update(Mat image) {
        if (trackers.isEmpty() || objects.isEmpty()) return false;
        boolean result = true;
        for (int i = 0; i < trackers.size(); i++) {
            TrackerKCF tracker = trackers.get(i);
            Rect rect = objects.get(i);
            result &= tracker.update(image, rect);
        }
        if (!result) clear();
        return result;
    }

    public List<Mat> getMatObjects() {
        List<Mat> listOfMats = new ArrayList<>();

        for (Rect rect : objects) {
            Mat mat = new Mat(5, 1, CV_32SC2);
            IntIndexer indexer = mat.createIndexer();

            int x = Math.round(rect.x());
            int y = Math.round(rect.y());
            int w = Math.round(rect.width());
            int h = Math.round(rect.height());

            indexer.put(0, 0, 0, x);
            indexer.put(0, 0, 1, y);
            indexer.put(1, 0, 0, x + w);
            indexer.put(1, 0, 1, y);
            indexer.put(2, 0, 0, x + w);
            indexer.put(2, 0, 1, y + h);
            indexer.put(3, 0, 0, x);
            indexer.put(3, 0, 1, y + h);
            indexer.put(4, 0, 0, x);
            indexer.put(4, 0, 1, y);

            listOfMats.add(mat);
        }
        return listOfMats;
    }

    private void clear(){
        trackers.clear();
        objects.clear();
    }
}
