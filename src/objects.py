from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


class Tune():
    """Tune library for audio files. Contain samples, sample rate and file name.
    """
    samples: np.ndarray
    sample_rate: int
    file_path: Path 

    def __init__(self,
                 samples: np.ndarray,
                 sample_rate: int,
                 file_path: Path, ) -> None:
        """Tune Object.

        Parameters
        ----------
        samples : np.ndarray
            Audio samples.
        sample_rate : int
            Sample rate
        file_path : Path
            Path where Tune was extract from.
        """
        self.samples = samples
        self.sample_rate = sample_rate
        self.file_path = file_path

    @property
    def time_length(self,) -> float:
        return self.samples.shape[0]/self.sample_rate
    
    def __repr__(self) -> str:
        return f'{self.time_length} seconds audio.'

class MelSGram():
    """ Mel Spectrogram. Contain file name and content. 
    """
    file_name: Path
    content: np.ndarray

    def __init__(self,    
                 file_name: Path,
                 content: np.ndarray,
                 sample_rate: int) -> None:
        """Mel Spectrogram object.

        Parameters
        ----------
        file_name : Path
            File name where Mel Spectrogram was extract from
        content : np.ndarray
            Mel Spectrogram contents as a 2 Dimensional matrix.
        sample_rate : int
            Sample rate
        """
        self.file_name = file_name
        self.content = content
        self.sample_rate = sample_rate

    @property
    def shape(self):
        return self.content.shape
    
    def __repr__(self) -> str:
        return f'Mel Spectrogram with shape {self.shape}'
    
    def plot(self,):
        """Plot mel spectrogram.
        """
        librosa.display.specshow(self.content, 
                                 sr=self.sample_rate,
                                 x_axis='time', 
                                 y_axis='mel')
        plt.title(self.file_name.name)
        plt.colorbar(format='%+2.0f dB')            