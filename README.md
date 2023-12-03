# NOTES

## DATA STUFF
Data available [here](https://downloads.khinsider.com/game-soundtracks/album/legend-of-zelda-the-a-link-to-the-past-snes)

[MP3 to WAV programatically, StackOverFlow](https://stackoverflow.com/questions/3049572/how-to-convert-mp3-to-wav-in-python)

## ON MEL SPECTROGRAM

[Ketanhdoshi, Github IO](https://ketanhdoshi.github.io/Audio-Mel/)
[Gartzman, Medium](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
[Mel Bands, StackOverFlow](https://stackoverflow.com/questions/62623975/why-128-mel-bands-are-used-in-mel-spectrograms)

## MODEL STUFF

[Autoencoder in Tensorflow](https://www.tensorflow.org/tutorials/generative/autoencoder)
[Deeper Autoencoder](https://blog.keras.io/building-autoencoders-in-keras.html)
[CVAE](https://www.tensorflow.org/tutorials/generative/cvae)

# On the data analysis

The file metdata.csv contain a classification I personally did on the audio samples. I classify the samples into two categories:
- soundeffect
- soundscape

Soundeffects are, in general, short tunes that play when we interact we something in game. I expand on that concept to consider also melodies played under certain circumstances, like boss fight, and cutscenes. Two examples of such _soundeffects_ are:
- _23. Priest of the Dark Order.wav_: this track plays when Agahnin 'kills' Zelda. Even being a long tune, it is consider a cutscene soundeffect;
- _01. Title ~ Link to the Past.wav_: this track is the menu theme. Then I considered it a soundeffect.

Likewise, the soundscapes are tunes played in exploration phases, in the open world or in a dungeon (or other closed spaces). It is important to note, though, some tunes are considered soundscapes even though they are short, like _02. Beginning of the Journey.wav_ or _08. Princess Zelda's Rescue.wav_. One of the reasons is because some of such tunes occurs in pairs, because an ambience (like rain or wind) is applied to it. I kept them in the dataset to check if they appear close in the embedded space. For instance, we have the pairs: _25. Black Mist.wav_ and _26. Black Mist (Storm).wav_, and _06. Majestic Castle.wav_ and _07. Majestic Castle (Storm).wav_.

# Data Zipping

In order to zip the mel-sgrams, run
```
$ zip -r ./data/processed/soundscapes-mel-sgrams.zip ./data/processed/soundscapes-mel-sgrams
```