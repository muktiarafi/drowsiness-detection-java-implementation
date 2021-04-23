package dev.muktiarafi.drowsinessdetection;

import javafx.scene.image.Image;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.ByteArrayInputStream;

public class CVUtils {
    private VideoCapture capture;
    private CascadeClassifier eyesCascade;
    private CascadeClassifier faceCascade;

    public void loadVideo(int deviceId) {
        capture = new VideoCapture(deviceId);
    }

    public void loadEyesCascade() {
        eyesCascade = new CascadeClassifier();
        eyesCascade.load("./src/main/resources/haarcascades/haarcascade_eye.xml");
    }

    public void loadFaceCascade() {
        faceCascade = new CascadeClassifier();
        faceCascade.load("./src/main/resources/haarcascades/haarcascade_frontalface_default.xml");
    }

    public Image matToImage(Mat mat) {
        MatOfByte bytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, bytes);
        ByteArrayInputStream inputStream = new ByteArrayInputStream(bytes.toArray());
        return new Image(inputStream);
    }

    public void detectFace(Mat inputMat) {
        MatOfRect facesDetected = new MatOfRect();
        int faceSize = Math.round(inputMat.rows() * 0.1f);
        faceCascade.detectMultiScale(
                inputMat,
                facesDetected,
                1.1,
                3,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(faceSize, faceSize),
                new Size()
        );
        Rect[] facesArray = facesDetected.toArray();
        for(Rect face : facesArray) {
            Imgproc.rectangle(inputMat, face.tl(), face.br(), new Scalar(0, 0, 255), 3 );
        }
    }

    public Mat detectEyes(Mat inputMat) {
        MatOfRect eyes = new MatOfRect();
        Mat eyesRoi = new Mat();
        eyesCascade.detectMultiScale(inputMat, eyes, 1.1, 4);
        Rect[] eyesArr = eyes.toArray();
        Mat gray = new Mat();
        Imgproc.cvtColor(inputMat, gray, Imgproc.COLOR_BGR2GRAY);
        for (Rect eye : eyesArr) {
            Mat roiGray = gray
                    .rowRange(eye.y, eye.y + eye.height)
                    .colRange(eye.x, eye.x + eye.width);
            Mat roiColor = inputMat
                    .rowRange(eye.y, eye.y + eye.height)
                    .colRange(eye.x, eye.x + eye.width);

            MatOfRect eyes2 = new MatOfRect();
            eyesCascade.detectMultiScale(roiGray, eyes2);
            Rect[] eyes2Arr = eyes2.toArray();
            if (eyes2Arr.length != 0) {
                for (Rect eye2 : eyes2Arr) {
                    eyesRoi = roiColor
                            .rowRange(eye2.y, eye2.y + eye2.height)
                            .colRange(eye2.x, eye2.x + eye2.width);
                }
            }
        }

        return eyesRoi;
    }

    public void putTextToMat(Mat inputMat, String text) {
        Imgproc.putText(
                inputMat,
                text,
                new Point(50, 50),
                Imgproc.FONT_HERSHEY_COMPLEX,
                3,
                new Scalar(0, 0, 255),
                2,
                Imgproc.LINE_4
        );
    }

    public double getCaptureWidth() {
        return capture.get(Videoio.CAP_PROP_FRAME_WIDTH);
    }

    public double getCaptureHeight() {
        return capture.get(Videoio.CAP_PROP_FRAME_HEIGHT);
    }

    public void releaseCapture() {
        capture.release();
    }

    public VideoCapture getCapture() {
        return capture;
    }
}
