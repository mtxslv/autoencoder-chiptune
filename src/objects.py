from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

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

    def pad(self, target_time: float) -> None:
        """Pad the audio to a target time by repeating the existing samples.

        Parameters
        ----------
        target_time : float
            Target duration of the padded audio in seconds.
        """
        current_time = self.time_length
        target_samples = int(target_time * self.sample_rate)

        if target_samples <= self.samples.shape[0]:
            # No padding needed, return
            return


        # Repeat the existing samples to pad the audio
        repeated_samples = np.tile(self.samples, 
                                   int(np.ceil(target_samples / self.samples.shape[0])))

        # Trim the repeated samples to the target length
        padded_samples = repeated_samples[:target_samples]

        # Update the samples attribute
        self.samples = padded_samples

    def crop(self, target_time: float) -> None:
        """Crop the audio to a target time by removing excess samples.

        Parameters
        ----------
        target_time : float
            Target duration of the cropped audio in seconds.
        """
        target_samples = int(target_time * self.sample_rate)

        if target_samples >= self.samples.shape[0]:
            # No cropping needed, return
            return

        # Crop the audio to the target length
        cropped_samples = self.samples[:target_samples]

        # Update the samples attribute
        self.samples = cropped_samples

    def dump(self, path: Path, output_type: str = 'numpy') -> None:
        """Dump the audio to a file in the specified format.

        Parameters
        ----------
        path : Path
            Output folder path.
        output_type : str, optional
            Type of output file ('numpy' or 'wav'), by default 'numpy'.
        """
        # Check if the specified path is a folder and exists
        if not path.is_dir() or not path.exists():
            raise ValueError("Invalid output folder path.")

        # Extract the file name from self.file_path
        file_name = self.file_path.stem

        # Create the full output file path
        output_file_path = path / f"{file_name}.{output_type}"

        if output_type == 'numpy':
            # Save the audio samples as a NumPy array
            np.save(output_file_path, self.samples)
        elif output_type == 'wav':
            # Save the audio samples as a WAV file
            wavfile.write(output_file_path, self.sample_rate, self.samples)
        else:
            raise ValueError("Invalid output_type. Supported types are 'numpy' or 'wav'.")        

class MelSGram():
    """ Mel Spectrogram. Contain file name and content. 
    """
    file_path: Path
    content: np.ndarray

    def __init__(self,    
                 file_path: Path,
                 content: np.ndarray,
                 sample_rate: int) -> None:
        """Mel Spectrogram object.

        Parameters
        ----------
        file_path : Path
            File path where Mel Spectrogram was extract from
        content : np.ndarray
            Mel Spectrogram contents as a 2 Dimensional matrix.
        sample_rate : int
            Sample rate
        """
        self.file_path = file_path
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

    def dump(self, path: Path) -> None:
        """Dump the Mel Spectrogram to a numpy file.

        Parameters
        ----------
        path : Path
            Output folder path.
        """
        # Check if the specified path is a folder and exists
        if not path.is_dir() or not path.exists():
            raise ValueError("Invalid output folder path.")

        # Extract the file name from self.file_path
        file_name = self.file_path.stem

        # Create the full output file path
        output_file_path = path / f"{file_name}.npy"

        if output_type == 'numpy':
            # Save the Mel Sgrams as a NumPy array
            np.save(output_file_path, self.content)