# Seismic_Data_Inpainting

Задача: реконструкция сейсмических кубов.  

Описание подхода к решению:

1. Были нарезаны данные из исходного куба ([dataset_slicing.ipynb](https://github.com/DanilKonon/Seismic_Data_Inpainting/blob/main/dataset_slicing.ipynb))
2. Был натренирован автоенкодер ([for_autoencdoer.ipynb](https://github.com/DanilKonon/Seismic_Data_Inpainting/blob/main/for_autoencoder.ipynb) )
3. Ну и в конце был натренирован Unet с perceptual и style лоссами, которые были вычислены с помощью прзнаковых карт автоенкодера. Для удобства постановки экспериментов код для тренировки финальной модели можно задать с помощью конфиг-файла. Пример [такого файла](https://github.com/DanilKonon/Seismic_Data_Inpainting/blob/main/example.json). 



Очень сильно помогли в данной работе два гитхаб репозитория: 

1. https://github.com/MathiasGruber/PConv-Keras откуда я взял реализацию лосс функций 
2. https://github.com/clovaai/AdamP реализация модифицированного метода обучения. хотя данный оптимизатор в итоге не вошёл в финальную версию модели. 



Очень сильно помогли и вдохновили статьи: 

1. [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
2. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

