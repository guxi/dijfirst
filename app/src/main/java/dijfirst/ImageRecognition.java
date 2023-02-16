package dijfirst;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;

public class ImageRecognition {
  private String modelUrl = "";

  public List<Item> irTopK_item(String modelurl, String modelName, String picurl, int topNum) {
    modelUrl = Paths.get(modelurl + "/" + modelName).toString();
    Classifications cfs = ir(modelName, picurl);

    List<Item> items = new ArrayList<Item>();
    for (int i = 0; i < topNum; i++) {
      var cf = cfs.topK(topNum).get(i);
      Item item = new Item();
      item.setClassname(cf.getClassName());
      item.setValue(cf.getProbability());
      items.add(item);
    }
    return items;
  }

  public String irTopK_String(String modelurl, String modelName, String picurl) {
    modelUrl = Paths.get(modelurl + "/" + modelName).toString();
    return ir(modelName, picurl).toString();
  }

  public String irTopK_Json(String modelurl, String modelName, String picurl) {
    modelUrl = Paths.get(modelurl + "/" + modelName).toString();
    return ir(modelName, picurl).toJson();
  }

  @SuppressWarnings("rawtypes")
  private Classifications ir(String modelName, String picurl) {
    try {
      if (!Files.exists(Paths.get(modelUrl)))
        download(modelName);

      Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
          .addTransform(new Resize(256))
          .addTransform(new CenterCrop(224, 224))
          .addTransform(new ToTensor())
          .addTransform(new Normalize(
              new float[] { 0.485f, 0.456f, 0.406f },
              new float[] { 0.229f, 0.224f, 0.225f }))
          .optApplySoftmax(true)
          .build();

      Criteria<Image, Classifications> criteria = Criteria.builder()
          .setTypes(Image.class, Classifications.class)
          .optModelPath(Paths.get(modelUrl))
          .optOption("mapLocation", "true") // this model requires mapLocation for GPU
          .optTranslator(translator)
          .optProgress(new ProgressBar()).build();

      ZooModel model = criteria.loadModel();

      var img = ImageFactory.getInstance().fromUrl(picurl);
      img.getWrappedImage();

      Predictor<Image, Classifications> predictor = model.newPredictor();
      Classifications classifications = predictor.predict(img);
      return classifications;
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  private void download(String name) throws Exception {
    String url_pt = "";
    String url_txt = "";
    String save_pt = "";
    String save_txt = "";

    switch (name) {
      case "resnet18": {
        url_pt = "https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/resnet/0.0.1/traced_resnet18.pt.gz";
        url_txt = "https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/synset.txt";
        save_pt = "resnet18.pt";
        save_txt = "synset.txt";
        break;
      }
      case "":
    }

    DownloadUtils.download(
        url_pt, modelUrl + "/" + save_pt, new ProgressBar());
    DownloadUtils.download(
        url_txt, modelUrl + "/" + save_txt, new ProgressBar());
  }
}
