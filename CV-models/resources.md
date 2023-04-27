We will be using libraries like huggingface and mmlab for building training and inference scripts for various models on a custom dataset.

For object detector or instance segmentor, we need to store the annotations in COCO format and these annotations can directly be used with detectron or mmlab libraries.

For semantic segmentation,we need the outputs stored as maps. These maps can have different formats by which they store the class of every pixel

To do : Image to text models.