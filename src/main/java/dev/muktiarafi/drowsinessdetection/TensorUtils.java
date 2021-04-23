package dev.muktiarafi.drowsinessdetection;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.tensorflow.*;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Objects;

public class TensorUtils {
    private TensorInfo inputTensor;
    private TensorInfo outputTensor;
    private Session session;

    public void loadModel() throws Exception {
        SavedModelBundle bundle = SavedModelBundle.load("./src/main/resources/model", "serve");
        MetaGraphDef metaGraphDef = MetaGraphDef.parseFrom(bundle.metaGraphDef());
        SignatureDef signatureDef = metaGraphDef.getSignatureDefMap().get("serving_default");
        inputTensor = signatureDef.getInputsMap()
                .values()
                .stream()
                .filter(Objects::nonNull)
                .findFirst()
                .orElseThrow(Exception::new);
        outputTensor = signatureDef.getOutputsMap()
                .values()
                .stream()
                .filter(Objects::nonNull)
                .findFirst()
                .orElseThrow(Exception::new);
        session = bundle.session();
    }

    public float predict(Mat mat) throws IOException {
        Tensor<Float> imageTensor = matToTensor(mat);
            Tensor<Float> result = session.runner()
                    .feed(inputTensor.getName(), imageTensor)
                    .fetch(outputTensor.getName()).run().get(0).expect(Float.class);
            float[] prediction = result.copyTo(new float[1][1])[0];
            return prediction[0];
    }

    private Tensor<Float> matToTensor(Mat mat) throws IOException {
        int imageWidth = mat.width();
        int imageHeight = mat.height();
        int channels = mat.channels();
        FloatBuffer fb = FloatBuffer.allocate(imageWidth * imageHeight * channels);
        BufferedImage bufferedImage = matToBufferedImage(mat);

        int index = 0;
        for (int row = 0; row < imageHeight; row++) {
            for (int column = 0; column < imageWidth; column++) {
                int pixel = bufferedImage.getRGB(column, row);
                float red = (pixel >> 16) & 0xff;
                float green = (pixel >> 8) & 0xff;
                float blue = pixel & 0xff;
                red = red / 255f;
                green = green / 255f;
                blue = blue / 255f;
                fb.put(index++, red);
                fb.put(index++, green);
                fb.put(index++, blue);
            }
        }

        return Tensor.create(new long[]{1, imageHeight, imageWidth, channels}, fb);
    }

    private BufferedImage matToBufferedImage(Mat mat) throws IOException {
        MatOfByte bytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, bytes);
        byte[] byteArray = bytes.toArray();
        InputStream in = new ByteArrayInputStream(byteArray);

        return ImageIO.read(in);
    }

    public void closeSession() {
        session.close();
    }
}
