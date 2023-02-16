package dijfirst;

import java.nio.file.Paths;
import java.util.Iterator;
import java.text.DecimalFormat;

/**
 * 
 */
public class App {
    public static void main(String[] args) {
        String modelurl = Paths.get(
                Paths.get("").toAbsolutePath()
                        + "/app/build/pytorch_models/")
                .toString();
        String picurl = Paths.get(modelurl).toString()
                + "/h2.jpg";

        ImageRecognition IR = new ImageRecognition();
        // System.out.println(IR.irTopK_String(modelurl, "resnet18", picurl));
        System.out.println(IR.irTopK_Json(modelurl, "resnet18", picurl));

        // Iterator<Item> it = IR.irTopK_item(modelurl, "resnet18", picurl,
        // 10).iterator();
        // DecimalFormat df = new DecimalFormat("#0.0000");
        // while (it.hasNext()) {
        // Item item = it.next();
        // System.out.println(item.getClassname());
        // System.out.println(df.format(item.getValue()));
        // }
    }
}
