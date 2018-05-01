# Weekly Schedule

### Week 1 - ML for Interaction

- What is Artificial Intelligence - Machine Learning?
- Overview
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Classification 1
  - Decision Tree
  - K-Nearest Neighbor
  - Intro to Wekinator
- Readings
  - Creative AI:  [https://medium.com/@creativeai/creativeai-9d4b2346faf3](https://medium.com/@creativeai/creativeai-9d4b2346faf3)
  - What is AI &amp; History: [Chapter 1 - Introduction - Artificial Intelligence a Modern Approach](http://web.cecs.pdx.edu/~mperkows/CLASS_479/2017_ZZ_00/02__GOOD_Russel=Norvig=Artificial%20Intelligence%20A%20Modern%20Approach%20(3rd%20Edition).pdf)

### Week 2 - More ML for Interaction

- Regression
  - Linear/Polynomial
  - Neural Net (soft intro)
- Classification Pt. 2
  - Naive Bayes
  - Ada Boost
  - Support Vector Machine
  - Probability
  - Multiple Classifiers
- Readings
  - AI Revolution Parts [1](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html) ~~&amp; [2](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-2.html)~~
  - Play [Paperclips](http://www.decisionproblem.com/paperclips/)

### Week 3 - Time Based Interactions

- ~~Dynamic Time Warping~~ (Moved to week 4)
- Various sensing methods
  - Kinect
  - Vision - OpenCV
    - Fiducials
    - Frame Differencing
    - Optical Flow
  - Audio
    - RMS
    - FFT
    - Spectral Centroid
    - Mel-Frequency Cepstral Coefficients
    - [Maximilian](http://maximilian.strangeloop.co.uk/)
    - [http://www.sonicvisualiser.org/](http://www.sonicvisualiser.org/)
- Readings
  - AI Revolution Parts [2](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-2.html)
  - ~~[Ethics of Machine Intelligence](http://us8.campaign-archive1.com/?u=bdb368b9a389b010c19dbcd54&amp;id=f2e0882b79)~~
  - ~~[Filtering Dissent](https://newleftreview.org/II/99/rodrigo-ochigame-james-holston-filtering-dissent)~~
  - ~~Breaking The Black Box - Parts [3](https://www.propublica.org/article/breaking-the-black-box-when-machines-learn-by-experimenting-on-us) &amp; [4](https://www.propublica.org/article/breaking-the-black-box-how-machines-learn-to-be-racist?word=Trump)~~
  - ~~[Facial Recognition](https://www.eff.org/deeplinks/2016/10/memo-doj-facial-recognitions-threat-privacy-worse-anyone-thought)~~

# Week 5 - _Interaction Project Due_

- Intro to Neural Networks
- The Perceptron
- Readings
  * [AI programs exhibit racial and gender biases](https://www.theguardian.com/technology/2017/apr/13/ai-programs-exhibit-racist-and-sexist-biases-research-reveals)
  * [Combating Bias](https://www.bloomberg.com/news/articles/2017-12-04/researchers-combat-gender-and-racial-bias-in-artificial-intelligence)

### Week 6 - Neural Nets from scratch: Math

- Linear Algebra Primer
  - Vectors
  - Matrices
- Activation Function
  - Sigmoid
- Calculus Primer
  - Derivatives
  - Power Rule
  - Chain Rule
- Gradient Descent
- Reading
  - Make Your Own Neural Network, Part 1 - How They Work, Tariq Rashid
- Other
  - [Moral Machine](http://moralmachine.mit.edu/)
- Readings
  - [Machine Ethics](http://www.nature.com/news/machine-ethics-the-robot-s-dilemma-1.17881)
  - [Speculative Design Chapter 1](http://readings.design/PDF/speculative-everything.pdf)

### Week 7  - Neural Nets from scratch: Code

- MNIST
- Python: Jupyter, IPython, Numpy, SciPy, etc.
- Reading
  - Deep Learning History: http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/

### SPRING BREAK

# Week 8 - _A.I. Speculative Design Project Due_


### Week 9 - More Neural Networks

- Activation Functions
	* binary step, linear, sigmoid, tanh, ReLU, softmax
- Optimization
	* stochastic/batch/minibatch, momentum, decay
	* adaptive methods
    	* adagrad, adadelta, rmsprop
	* adam
- Loss Functions
	* Square Error, Cross Entropy
- Intro Keras/Tensorflow 

### Week 10 - Convolution Neural Networks


- Architectures 
- Image Classification - [CIFAR-10](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py) ( [online demo](http://ml4a.github.io/demos/confusion_cifar/))
- [deep dream](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream)
- [style transfer](https://github.com/cysmith/neural-style-tf)
- [photo realistic style transfer](https://github.com/LouieYang/deep-photo-styletransfer-tf)

### Week 11 - Recurrent Neural Networks

- RNN
- LSTM
- Applications of RNN
- Generating Text
#### Sound
- [Magenta](https://magenta.tensorflow.org/) (RNN)
  - [drum patterns](https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn)
  - [melody](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn)
  - [polyphony](https://github.com/tensorflow/magenta/tree/master/magenta/models/polyphony_rnn)
  - [improvisation](https://github.com/tensorflow/magenta/tree/master/magenta/models/improv_rnn)
  - [dynamics](https://github.com/tensorflow/magenta/tree/master/magenta/models/performance_rnn)
- [wavenet](https://github.com/ibab/tensorflow-wavenet) (CNN)
- [nsynth](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth) ( [https://experiments.withgoogle.com/ai/sound-maker](https://experiments.withgoogle.com/ai/sound-maker))
- [Organizing sound](https://github.com/kylemcdonald/AudioNotebooks)
  - t-sne:  [https://experiments.withgoogle.com/ai/drum-machine/view/](https://experiments.withgoogle.com/ai/drum-machine/view/)
  - bird sounds:  [https://experiments.withgoogle.com/ai/bird-sounds/view/](https://experiments.withgoogle.com/ai/bird-sounds/view/)
- Commercial
  - [http://www.aiva.ai/](http://www.aiva.ai/)
  - [https://www.ampermusic.com/](https://www.ampermusic.com/)
- Music Information Retrieval
  - [https://github.com/craffel/mir\_eval](https://github.com/craffel/mir_eval)
- Voice
  - [https://lyrebird.ai/](https://lyrebird.ai/)
- Reading
  - [https://medium.com/artists-and-machine-intelligence/neural-nets-for-generating-music-f46dffac21c0](https://medium.com/artists-and-machine-intelligence/neural-nets-for-generating-music-f46dffac21c0)
- Readings
  - [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [https://arstechnica.com/gaming/2016/06/an-ai-wrote-this-movie-and-its-strangely-moving/](https://arstechnica.com/gaming/2016/06/an-ai-wrote-this-movie-and-its-strangely-moving/)

### Week 12 - Generative Models


* [PCA Faces](https://github.com/ml4a/ml4a-guides/blob/master/notebooks/eigenfaces.ipynb)
* [VAE](https://github.com/jmetzen/jmetzen.github.com/blob/source/content/notebooks/vae.ipynb)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [BEGAN](https://github.com/carpedm20/BEGAN-tensorflow)
* [Progressive Growth GAN](https://github.com/tkarras/progressive_growing_of_gans)
* [InfoGAN](https://github.com/JonathanRaiman/tensorflow-infogan)
* [DiscoGan](https://github.com/carpedm20/DiscoGAN-pytorch) (pytorch)
* [StackedGAN](https://github.com/hanzhanggit/StackGAN)
* [GANGogh](https://github.com/rkjones4/GANGogh)
* [DGN](https://github.com/Evolving-AI-Lab/synthesizing) (caffe)
* [CPPN/VAE/GAN](https://github.com/hardmaru/cppn-gan-vae-tensorflow) 
* [RESNET/CPPN/GAN](https://github.com/hardmaru/resnet-cppn-gan-tensorflow)
* [pix2pix](https://github.com/memo/pix2pix-tensorflow)
* [CycleGAN](https://github.com/xhujoy/CycleGAN-tensorflow)
* [Deep Painterly Harmonization](https://github.com/luanfujun/deep-painterly-harmonization) (torch)  


### Week 13 - Final Project Work

# Week 14 - _Final Project Due_
