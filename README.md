# Welcome! 👋👋

This project demonstrates how to visualize a set of songs in a 2D plane. It does so by first extracting the mel-spectrograms of the audios and then passing them into a Convolution Auto Encoder with a latent space (read bottleneck layer) containing two axis (i.e., neurons) only. Of course, more stuff is done in the mean time so if you want to know more read such details below, ok?

> oh, btw... this is a pet project, so take it easy.  

# Data Processing

## 🎶 Dataset 🎶

Data is available [here](https://downloads.khinsider.com/game-soundtracks/album/legend-of-zelda-the-a-link-to-the-past-snes). I downloaded one by one by hand. 😮‍💨

Once [locally available](/data/raw/mp3/), I converted the files to the [`.wav` format](/data/raw/wav/). This is done in the [convert_data_to_wav.ipynb](/notebooks/convert_data_to_wav.ipynb) jupyter file.

> Naturally, I should have turned it into an utils function, but nah.🙄

I did a lil' bit of 'exploratory data analysis' on [eda.ipynb](/notebooks/eda.ipynb). It was necessary because I was unsure how to feed different length audios into an autoencoder network. In the end, the question became 'which ones to feed'. This caused me to write the [metadata.csv](/data/metadata.csv) files, where I classify the tune as 'soundeffect' or 'soundscape'. More details below 😉

### 📊 On the data analysis 📊

The file metdata.csv contain a classification I personally did on the audio samples. I classify the samples into two categories:
- soundeffect
- soundscape

Soundeffects are, in general, short tunes that play when we interact we something in game. I expand on that concept to consider also melodies played under certain circumstances, like boss fight, and cutscenes. Two examples of such _soundeffects_ are:
- _23. Priest of the Dark Order.wav_: this track plays when Agahnin 'finishes off' Zelda. Even being a long tune, it is consider a cutscene soundeffect;
- _01. Title ~ Link to the Past.wav_: this track is the menu theme. Then I considered it a soundeffect.

Likewise, the soundscapes are tunes played in exploration phases, in the open world or in a dungeon (or other closed spaces). It is important to note, though, some tunes are considered soundscapes even though they are short, like _02. Beginning of the Journey.wav_ or _08. Princess Zelda's Rescue.wav_. One of the reasons is because some of such tunes occurs in pairs, because an ambience (like rain or wind) is applied to it. I kept them in the dataset to check if they appear close in the embedded space. For instance, we have the pairs: _25. Black Mist.wav_ and _26. Black Mist (Storm).wav_, and _06. Majestic Castle.wav_ and _07. Majestic Castle (Storm).wav_.

## ✂️🖇️ Preprocessing 🖇️✂️

As seen in [processing.ipynb](/notebooks/processing.ipynb), I pad and crop the soundscapes. My logic was like: 
> "Take the distribution of audio lengths. I'll find a good point to crop/pad the files. Well, it looks like padding by copying and pasting over and over is not a big deal, since people on the internet post [videos with 10 hours each on youtube](https://youtu.be/pLgERUpr40A?si=Y94AFurNDrqVKmmr) and it sounds great (I took the idea of padding by repeating from it). Well, cropping at the 75% quartile looked OK for me because the longest file would lose only 25% of its contents at most and padding the rest was considered fine from design. The code I used to do that was defined at [objects.py](/src/objects.py) and [dataset.py](/src/dataset.py) based on what I learned about Clean Architecture (domain objects, you know?). Of course, some unit testing would make it even prettier, but I did not care at the time, hee hee hee"

The products of preprocessing can be foun in the [processed/ folder](/data/processed/). 

I then zipped the mel-specs with this command

```
$ zip -r ./data/processed/soundscapes-mel-sgrams.zip ./data/processed/soundscapes-mel-sgrams
```

and uploaded the resulting [soundscapes-mel-sgrams.zip](/data/processed/soundscapes-mel-sgrams.zip) file to my GDrive. 

## 🧮📉 Running the autoencoder 📉🧮

The [autoencoder.ipynb](/notebooks/autoencoder.ipynb) contains the steps to generate the embedded space's coordinates for each tune. Basically I unzip the data, pad it to make a even-dimensions-tensor, define a Convolutional Autoencoder, fine tune it and extract the embedded (latent) dimensions.

> I actually played a lot with the autoencoder class. It is a collage from several sources (you can check them out below on the Ref.). I tried to add more convolutional layers both in the encoder and decoder modules, but this ended by messing up the output over and over again. I then moved on and just left the quantity of filters in the conv layers as adjustable parameters. This 'version' of the class is the one seen at [models.py](/src/models.py).

I trained this network, which took like 36 minutes approx. The last step was then to extract the latent-space dimensions. The result is in the [embeddings.json](/data/processed/embeddings.json) file.
 
## 🔭 Visualizing 🔭

To see the results, check the [visualizing.ipynb](/notebooks/visualizing.ipynb) file.

# 📚 References 📚

## MP3 to Wav

- ["How to convert MP3 to WAV in Python", by yydl on StackOverFlow](https://stackoverflow.com/questions/3049572/how-to-convert-mp3-to-wav-in-python)

## On Mel Spectrograms

- [Audio Deep Learning Made Simple - Why Mel Spectrograms perform better, by Ketan doshi on Github IO](https://ketanhdoshi.github.io/Audio-Mel/)
- [Getting to Know the Mel Spectrogram, by Dalya Gartzman on Medium](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
- ["Why 128 mel bands are used in mel spectrograms?",  by swe87 on StackOverFlow](https://stackoverflow.com/questions/62623975/why-128-mel-bands-are-used-in-mel-spectrograms)

## On Autoencoder

- [Intro to Autoencoders, by Tensorflow](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [Building Autoencoders in Keras, by Francois Chollet in The Keras Blog](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Convolutional Variational Autoencoder, by Tensorflow](https://www.tensorflow.org/tutorials/generative/cvae)
