# quinoa

The rising interest in quinoa is due to its high protein content and gluten-free condition; nonetheless, the presence of foreign bodies in quinoa processing facilities is an issue that must be addressed. As a result, convolutional networks have been adopted, mostly because of their data extraction capabilities, which had not been utilized before for this purpose. Consequently, the main objective of this work is to evaluate convolutional networks with a learning transfer for foreign bodies identification in quinoa samples. For experimentation, quinoa samples were collected and manually split into 17 classes: quinoa grains and sixteen foreign bodies. Then, one thousand images were obtained from each class in RGB space and transformed into four different color spaces (L*a*b*, HSV, YCbCr, and Gray). Three convolutional networks (Alextnet, MobileNetv2, and DenseNet-201) were trained using the five color spaces, and the evaluation results were expressed in terms of accuracy and F-score. 

The functions implemented here allow training convolutional networks through different data sets of quinoa images. The images are found in their respective folders, for each one of the classes and color spaces. Likewise, they are resized in 224 x 224 and 227 x 227 according to the needs of each network. The links to the data are:

  https://drive.google.com/drive/folders/18RZ_HuW0g83UWATx_fhT0BhxWa2kxPp2?usp=sharing
  
  https://drive.google.com/drive/folders/14-cPUHWBW3E07Bl9eleFRMSD6zlN18Xp?usp=sharing 
