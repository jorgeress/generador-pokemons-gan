# Pokémon GAN: Generación de Criaturas mediante Deep Learning

Este proyecto utiliza una red **DCGAN (Deep Convolutional Generative Adversarial Network)** para sintetizar imágenes de Pokémon desde cero. El modelo ha sido entrenado durante **10,000 épocas** para aprender las características visuales, colores y formas del dataset original.



## Características del Proyecto
- **Arquitectura:** DCGAN implementada en TensorFlow/Keras.
- **Dataset:** ~800 imágenes de Pokémon (64x64 píxeles).
- **Optimización:** Implementación de TTUR (Two-Time-Scale Update Rule) para estabilidad.
- **Resultados:** Generación de sprites originales con coherencia cromática y morfológica.

## Desafíos Técnicos y Soluciones
Durante el entrenamiento, se enfrentaron retos clásicos de las redes generativas:

1. **Mode Collapse:** Se solucionó reduciendo el *learning rate* del Discriminador ($2 \times 10^{-5}$) frente al del Generador ($1 \times 10^{-4}$), permitiendo una exploración más rica del espacio latente.
2. **Checkerboard Artifacts:** Ajuste de filtros y strides en las capas `Conv2DTranspose` para suavizar las texturas.
3. **Estabilidad:** Uso de **Label Smoothing (0.9)** para evitar que el Discriminador se volviera demasiado dominante.

## 📊 Evolución del Entrenamiento
| Época 20 | Época 100 | Época 1000 |
| :---: | :---: | :---: |
| ![E20](Epochs/epoch_00020.png) | ![E100](Epochs/epoch_00100.png) | ![E1000](Epochs/epoch_01000.png) |


## Cómo usar el modelo
Si quieres generar tus propios Pokémon, puedes cargar el modelo pre-entrenado:

```python
import tensorflow as tf
generator = tf.keras.models.load_model('modelo/generador_pokemon_10000_final.h5')
noise = tf.random.normal([1, 100])
pokemon = generator(noise, training=False)
```

📚 Referencias
Dataset: Kaggle - Pokemon Images Dataset

Paper original: Unsupervised Representation Learning with DCGANs


