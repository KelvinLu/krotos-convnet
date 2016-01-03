# krotos-convnet
Experiments with songs, convolutional neural networks, and latent features &mdash; in the name of musical hipster-dom!

A pet project for learning how to use TensorFlow and sating my interests in "deep" (relative to a college student laptop's processing powers) learning and an old teenage fascination with music production (oh, so many hours spent in a pirated copy of FL Studio!)

Most of the work and methodology here coincides with Van den Oord, Dieleman, and Schrauwen's work in their paper, [Deep content-based music recommendation](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf), but toned down a few notches.



### #goals

We aim to build a CNN (and the surrounding data infrastructure) that will learn content-based features in the task of describing audio samples. Using a subset of the [Millon Song Dataset](http://labrosa.ee.columbia.edu/millionsong/), we hope to train the CNN to predict human-sourced [Last.fm tags](http://labrosa.ee.columbia.edu/millionsong/lastfm) of several hundred-thousand track samples sourced from [7digital](http://labrosa.ee.columbia.edu/millionsong/pages/tasks-demos#preview) with the task of feature engineering.

We will then fine tune our CNN model to predict latent feature vectors derived from the [Echo Nest Taste Profile](http://labrosa.ee.columbia.edu/millionsong/tasteprofile). This project will also provide an implementation to perform the collaborative filtering task that informs the audio's latent features.

In the far future, we might construct a web service to gauge musical compatibility for SoundCloud users, based on their likes. The cold-start problem described by Van de Oord et al. and Dieleman is very related to the nature of the SoundCloud platform. It would be hard to apply collaborative filtering findings, informed by "mainstream music", to "indie" artists that release their lesser-known music on SoundCloud.

> ... I've always wanted to say "check out my mixtape, fam" seriously.



## Frameworks, libraries, and services used

(This is a tentative list)

- Modelling
  - TensorFlow
- Processing
  - NumPy, SciPy
  - [LibROSA](http://bmcfee.github.io/librosa/index.html)
  - scikits.samplerate
  - [libsamplerate](http://www.mega-nerd.com/SRC/)
- Data Store
  - SQLite
  - HDF5
- Web service
  - Flask
  - Redis (as a job queue)
- Developer APIs
  - 7digital
  - SoundCloud



## Inspirations and citations

- Sander Dieleman on his work at Spotify, [our main source of inspiration](http://benanne.github.io/2014/08/05/spotify-cnns.html)

- Van den Oord, Aaron, Sander Dieleman, and Benjamin Schrauwen. ["Deep content-based music recommendation."](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf) Advances in Neural Information Processing Systems. 2013.

- Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere.
["The Million Song Dataset."](http://ismir2011.ismir.net/papers/OS6-1.pdf) ISMIR. 2011.

- McFee, Brian, et al. ["The million song dataset challenge."](http://eceweb.ucsd.edu/~gert/papers/msdc.pdf) Proceedings of the 21st international conference companion on World Wide Web. ACM, 2012.

- Law, Edith, et al. ["Evaluation of Algorithms Using Games: The Case of Music Tagging."](http://ismir2009.ismir.net/proceedings/OS5-5.pdf) ISMIR. 2009.

- Dieleman, Sander, and Benjamin Schrauwen. ["End-to-end learning for music audio."](https://dl.dropboxusercontent.com/u/19706734/paper_pt.pdf) Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014.

- Hu, Yifan, Yehuda Koren, and Chris Volinsky. ["Collaborative filtering for implicit feedback datasets."](http://yifanhu.net/PUB/cf.pdf) Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. IEEE, 2008.
