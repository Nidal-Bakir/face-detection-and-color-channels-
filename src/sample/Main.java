package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.opencv.core.*;

public class Main extends Application {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {

        FXMLLoader loader = new FXMLLoader(getClass().getResource("sample.fxml"));
        Parent root = loader.load();
        Controller controller = loader.getController();

        primaryStage.setTitle("Face Detection and Color Channels (BGR) ");
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.show();
        primaryStage.setOnCloseRequest(event -> {
            controller.setClosed();
        });
    }


    public static void main(String[] args) {

        launch(args);


    }
}
