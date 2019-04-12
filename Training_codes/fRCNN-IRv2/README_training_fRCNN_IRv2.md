# Training Codes
Authors: Shuichiro Komatsu, Takahiro Oike

Provides programs for training deep learning models. These programs are described in our paper, "Deep learning-assisted literature mining for in vitro radiosensitivity data".

![overview_figure1](https://github.com/shuichirokomatsu/Literature_Mining-Radiosensitivity/blob/master/Figure1.png)

## fRCNN-IRv2
You need to create TRrecord.
```
python create_tf_record.py \
  --annotations_dir=/tensorflow/data/annotations \
  --images_dir=/tensorflow/data/images \
  --output_dir=/tensorflow/data/ \
  --label_map_path=/tensorflow/data/label_map.pbtxt
```

Download pre-trained model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.
Run the next code to train, and you can achieved your own model.

```
python object_detection/legacy/train.py \
--train_dir=CP \
--pipeline_config_path=pipeline.config
```