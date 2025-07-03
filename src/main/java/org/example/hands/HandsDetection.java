package org.example.hands;


import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.example.utils.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.*;


public class HandsDetection {
    private final File palmModelPath = ResourceUtils.extractToTempFile("models/palm_detection_barracuda.onnx");

    private final OrtEnvironment envPalmModel = OrtEnvironment.getEnvironment();
    private final OrtSession sessionPalmModel = envPalmModel.createSession(palmModelPath.getAbsolutePath(), new OrtSession.SessionOptions());
    private final String inputNamePalmModel = sessionPalmModel.getInputNames().iterator().next();

    private final File postProcessingModelPath = ResourceUtils.extractToTempFile("models/PDPostProcessing_top2.onnx");

    private final OrtEnvironment envPostProcessingModel = OrtEnvironment.getEnvironment();
    private final OrtSession sessionPostProcessingModel = envPostProcessingModel.createSession(postProcessingModelPath.getAbsolutePath(), new OrtSession.SessionOptions());
    private final String inputNamePostProcessingModel = sessionPostProcessingModel.getInputNames().iterator().next();

    private final List<Anchor> anchors = new ArrayList<>();

    public HandsDetection() throws IOException, OrtException {
    }

    private void generateAnchors(Mat image) {
        ResizeInfo resizeInfo = resizeMat(image);
        int xOffset = resizeInfo.xOffset;
        int yOffset = resizeInfo.yOffset;
        float scale = resizeInfo.scale;

        anchors.clear();

        int[] featureMapSizes = {8, 16, 16, 16};
        float[] strides = {32f, 16f, 8f, 4f};
        float[][][] anchorSizes = {
                {{0.04f, 0.05f}, {0.05f, 0.06f}},
                {{0.08f, 0.10f}, {0.10f, 0.12f}},
                {{0.16f, 0.20f}, {0.20f, 0.25f}},
                {{0.32f, 0.40f}, {0.40f, 0.50f}}
        };

        for (int level = 0; level < featureMapSizes.length; level++) {
            int size = featureMapSizes[level];
            float stride = strides[level];

            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    // Центр якоря в пикселях 128×128 (с учётом stride)
                    float xCenter128 = (x + 0.5f) * stride;
                    float yCenter128 = (y + 0.5f) * stride;

                    // Применяем xOffset и yOffset — сдвиг паддинга, если есть
                    xCenter128 += xOffset;
                    yCenter128 += yOffset;

                    // Переводим в нормализованные координаты [0, 1]
                    float xCenterNorm = xCenter128 / 128f;
                    float yCenterNorm = yCenter128 / 128f;

                    for (float[] wh : anchorSizes[level]) {
                        float w = wh[0];
                        float h = wh[1];
                        anchors.add(new Anchor(xCenterNorm, yCenterNorm, w, h));
                    }
                }
            }
        }
    }


    public List<Mat> detect(Mat image) throws OrtException {
        Rect handBox = detectHandBox(image);
        if (handBox == null) return new ArrayList<>();
        Mat mat = new Mat(5, 1, CV_32SC2);
        IntIndexer indexer = mat.createIndexer();

        int x = handBox.x();
        int y = handBox.y();
        int w = handBox.width();
        int h = handBox.height();

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

        List<Mat> listOfMats = new ArrayList<>();
        listOfMats.add(mat);
        return listOfMats;
    }

    private ResizeInfo resizeMat(Mat mat) {
        int targetSize = 128;
        int origWidth = mat.cols();
        int origHeight = mat.rows();

        float scale = (float) targetSize / Math.max(origWidth, origHeight);
        int newWidth = Math.round(origWidth * scale);
        int newHeight = Math.round(origHeight * scale);

        Mat resized = new Mat();
        opencv_imgproc.resize(mat, resized, new Size(newWidth, newHeight));

        Mat output = new Mat(targetSize, targetSize, mat.type(), new Scalar(0, 0, 0, 0));
        int xOffset = (targetSize - newWidth) / 2;
        int yOffset = (targetSize - newHeight) / 2;

        Rect roi = new Rect(xOffset, yOffset, newWidth, newHeight);
        Mat submat = output.apply(roi);
        resized.copyTo(submat);
        return new ResizeInfo(output, scale, xOffset, yOffset);
    }

    public Rect detectHandBox(Mat image) throws OrtException {
        int H = image.rows(), W = image.cols();


        ResizeInfo output = resizeMat(image);
        Mat resized = output.mat;
        resized.convertTo(resized, CV_32F, 1.0 / 127.5, -1.0);


        FloatBuffer buffer = FloatBuffer.allocate(3 * 128 * 128);
        FloatIndexer indexer = resized.createIndexer();

        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < 128; y++) {
                for (int x = 0; x < 128; x++) {
                    float val = indexer.get(y, x, c);
                    buffer.put(val);
                }
            }
        }
        buffer.rewind();

        OnnxTensor input = OnnxTensor.createTensor(envPalmModel, buffer, new long[]{1, 3, 128, 128});

        try (OrtSession.Result resultPalmModel = sessionPalmModel.run(Map.of(inputNamePalmModel, input))) {
            float[][][] scoresPalmModel = (float[][][]) resultPalmModel.get(1).getValue();
            float[][][] boxesPalmModel = (float[][][]) resultPalmModel.get(0).getValue();

            OnnxTensor tensorScores = OnnxTensor.createTensor(envPostProcessingModel, scoresPalmModel);
            OnnxTensor tensorBoxes = OnnxTensor.createTensor(envPostProcessingModel, boxesPalmModel);

            try (OrtSession.Result resultPostProcessingModel = sessionPostProcessingModel.run(Map.of("classificators", tensorScores, "regressors", tensorBoxes))){

                float[][] detections = (float[][]) resultPostProcessingModel.get(0).getValue();

                // Обработка результатов детекции
                if (detections.length == 0 || detections[0].length < 8) {
                    return null;
                }

                // Берем первую (наиболее уверенную) детекцию
                float[] bestDetection = detections[0];
                float score = bestDetection[0];

                // Фильтр по confidence
                if (score < 0.5f) {
                    return null;
                }

                // Преобразование нормализованных координат в пиксельные
                float cx = bestDetection[1] * W;
                float cy = bestDetection[2] * H;
                float width = bestDetection[3] * W;
                float height = width; // Квадратный bounding box

                // Координаты углов прямоугольника
                int x1 = Math.round(cx - width/2);
                int y1 = Math.round(cy - height/2);
                int x2 = Math.round(cx + width/2);
                int y2 = Math.round(cy + height/2);

                // Проверка границ изображения
                x1 = Math.max(0, Math.min(x1, W-1));
                y1 = Math.max(0, Math.min(y1, H-1));
                x2 = Math.max(0, Math.min(x2, W-1));
                y2 = Math.max(0, Math.min(y2, H-1));

                return new Rect(x1, y1, x2 - x1, y2 - y1);
            }
        }
    }


    private record Anchor(float xCenter, float yCenter, float w, float h) {}

    private record ResizeInfo(Mat mat, float scale, int xOffset, int yOffset) {}

}
