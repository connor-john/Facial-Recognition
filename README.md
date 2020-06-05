# Facial-Recognition
 An implementation of Siamese Neural Networks for facial recognition

### Data

Using [yale face dataset](http://vision.ucsd.edu/content/yale-face-database)
_Collection of greyscale faces of differing light and facial expressions_

_example:_

<img src ="plots/data_example.png" width = 200>

### Siamese Networks
 Contrastive loss implementation

 $d=||f(x_1) - f(x_2)||_2$

 $L = yd^2 + (1 - y)[max(m - d,0)]^2$