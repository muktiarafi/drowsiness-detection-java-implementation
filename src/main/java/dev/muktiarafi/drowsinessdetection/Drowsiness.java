package dev.muktiarafi.drowsinessdetection;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import uk.co.caprica.vlcj.player.component.AudioPlayerComponent;

import java.io.IOException;

public class Drowsiness extends Application {
    public void start(Stage stage) throws Exception {
        OpenCV.loadShared();

        TensorUtils tensorUtils = new TensorUtils();
        CVUtils cvUtils = new CVUtils();
        tensorUtils.loadModel();
        cvUtils.loadVideo(0);
        cvUtils.loadFaceCascade();
        cvUtils.loadEyesCascade();

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                cvUtils.releaseCapture();
                tensorUtils.closeSession();
            }
        });

        String awakenPath = "./src/main/resources/awaken.mp3";
        AudioPlayerComponent audioPlayerComponent = new AudioPlayerComponent();
        audioPlayerComponent.mediaPlayer().media().startPaused(awakenPath);
        ImageView imageView = new ImageView();
        HBox hbox = new HBox(imageView);

        double captureWidth = cvUtils.getCaptureWidth();
        double captureHeight = cvUtils.getCaptureHeight();
        Scene scene = new Scene(hbox, captureWidth, captureHeight);

        stage.setScene(scene);
        stage.show();

        new AnimationTimer(){
            @Override
            public void handle(long l) {
                Mat frame = new Mat();
                cvUtils.getCapture().read(frame);
                Mat eyesRoi = cvUtils.detectEyes(frame);
                String status = "Eyes not detected";
                try {
                    if (!eyesRoi.empty()) {
                        Mat finalImage = new Mat();
                        Imgproc.resize(eyesRoi, finalImage, new Size(224, 224));
                        float predictions = tensorUtils.predict(finalImage);
                        if (predictions > 0.5) {
                            status = "Great you're awake!";
                            audioPlayerComponent.mediaPlayer().controls().stop();
                        } else {
                            status = "Are you sleepy?";
                            audioPlayerComponent.mediaPlayer().controls().play();
                        }
                    } else {
                        System.out.println("Eyes not detected");
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                cvUtils.detectFace(frame);
                cvUtils.putTextToMat(frame, status);
                Image image = cvUtils.matToImage(frame);
                imageView.setImage(image);
            }
        }.start();
    }

    public static void main(String[] args) {
        Application.launch(args);
    }
}
