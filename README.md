# magnatagatune-convnet (krotos)
Experiments with songs and convolutional neural networks.

A pet project for learning how to use TensorFlow and sating my interests in "deep" (relative to a college student laptop's processing powers) learning.



### tl:dr;

We aim to build a CNN (and the surrounding data infrastructure) that will learn content-based features in the task of describing audio samples. Using a subset of the [Millon Song Dataset](http://labrosa.ee.columbia.edu/millionsong/), we hope to train the CNN to predict human-sourced [Last.fm tags](http://labrosa.ee.columbia.edu/millionsong/lastfm) of several thousand track samples sourced from [7digital](http://labrosa.ee.columbia.edu/millionsong/pages/tasks-demos#preview).

We will then construct a web service to gauge musical compatibility for SoundCloud users, based on their likes.



## Frameworks, libraries, and services used

(This is a tentative list)

- Modelling and processing
  - TensorFlow
  - NumPy, SciPy
  - [LibROSA](http://bmcfee.github.io/librosa/index.html)
  - scikits.samplerate
- Web service
  - Flask
  - Redis
  - gevent
- APIs
  - SoundCloud's API
- Packages
  - [librsamplerate](http://www.mega-nerd.com/SRC/)



## Inspiration and citations

- Sander Dieleman on his work at Spotify, [our main source of inspiration](http://benanne.github.io/2014/08/05/spotify-cnns.html)
- Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
["The Million Song Dataset."](http://ismir2011.ismir.net/papers/OS6-1.pdf) ISMIR. 2011.
- Law, Edith, et al. ["Evaluation of Algorithms Using Games: The Case of Music Tagging."](http://ismir2009.ismir.net/proceedings/OS5-5.pdf) ISMIR. 2009.
- Van den Oord, Aaron, Sander Dieleman, and Benjamin Schrauwen. ["Deep content-based music recommendation."](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf) Advances in Neural Information Processing Systems. 2013.
- Dieleman, Sander, and Benjamin Schrauwen. ["End-to-end learning for music audio."](https://dl.dropboxusercontent.com/u/19706734/paper_pt.pdf) Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014.
