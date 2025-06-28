package org.example;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class CameraTracking {
    private final OpenCVFrameGrabber grabber;


    public CameraTracking(int deviceNumber) throws FrameGrabber.Exception {
        grabber = new OpenCVFrameGrabber(deviceNumber);
        grabber.start();
    }

    public Frame grab() throws FrameGrabber.Exception {
        return grabber.grab();
    }

    public void stop() throws FrameGrabber.Exception {
        grabber.stop();
    }
}
