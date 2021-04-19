# HairSegmentation
Разработка модели для сегментации волос, которая позволяет перекрашивать их в любой цвет, опираясь на предсказанную маску. В качестве модели используется сеть HairMatteNet, архитектура которой описана в статье [Real-time deep hair matting on mobile devices](https://arxiv.org/pdf/1712.07168.pdf).

## Примеры работы на тестовых данных
Картинки расположены в следующем порядке: исходное изображение, ground truth маска, предсказанная маска, результат окрашивания с помощью ground truth маски и результат окрашивания с помощью предсказанной маски. \
<img src="https://github.com/NastyaMittseva/HairSegmentation/blob/master/examples/99_epoch_test_results.jpg" width="70%" height="70%">

## Примеры инференса
Здесь представлены исходные гифки и гифки с перекрашенными волосами. \
<img src="https://github.com/NastyaMittseva/HairSegmentation/blob/master/examples/man.gif" width="35%" height="35%"> <img src="https://github.com/NastyaMittseva/HairSegmentation/blob/master/examples/result_man.gif" width="35%" height="35%">
<img src="https://github.com/NastyaMittseva/HairSegmentation/blob/master/examples/woman.gif" width="35%" height="35%"> <img src="https://github.com/NastyaMittseva/HairSegmentation/blob/master/examples/result_woman.gif" width="35%" height="35%"> 