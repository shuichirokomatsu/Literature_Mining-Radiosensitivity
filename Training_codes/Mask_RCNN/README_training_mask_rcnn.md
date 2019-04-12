# Training Codes
Authors: Shuichiro Komatsu, Takahiro Oike

Provides programs for training deep learning models. These programs are described in our paper, "Deep learning-assisted literature mining for in vitro radiosensitivity data".

![overview_figure1](https://github.com/shuichirokomatsu/Literature_Mining-Radiosensitivity/blob/master/Figure1.png)

## Mask RCNN
Move "create_mask_rcnn_tf_record.py" under "models/research/object_detection/dataset_tools" floder. 
Prepare pre-trained model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md , and put it under the "pre_trained_model" folder. 
Run the next codes, and you can achieved your own model.

```
python object_detection/dataset_tools/create_mask_rcnn_tf_record.py \
--data_dir=/train \
--annotations_dir=/train/annotations \
--image_dir=/train/images \
--output_dir=train.record \
--label_map_path=label.pbtxt
```

```
python object_detection/legacy/train.py \
--train_dir=CP \
--pipeline_config_path=pipeline.config
```

```
python object_detection/export_inference_graph.py \
--input_type=image_tensor \
--pipeline_config_path=pipeline.config \
--trained_checkpoint_prefix=/content/DL/CP/model.ckpt-[your epochs] \
--output_directory=IG
```