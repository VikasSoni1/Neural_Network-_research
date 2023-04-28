We will be using libraries like huggingface and mmlab for building training and inference scripts for various models on a custom dataset.

Image classification models uses a backbone as feature extractor and fully connected layers at top.

For object detector or instance segmentor, we need to store the annotations in COCO format and these annotations can directly be used with detectron or mmlab libraries.

For semantic segmentation,we need the outputs stored as maps. These maps can have different formats by which they store the class of every pixel


Image captioning is an encode-decoder model. Encoder is a CNN backbone and decoder is an RNN with Attention.

Depth estimation is just like a segmentation and use an encoder-decoder model with depth as prediction.

Text to image like models are generative models. Ouput of the generative network is controlled by encoding of the text or prompt using some RNN model.


To do : Image to text models.