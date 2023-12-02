from pathlib import Path

import librosa
import numpy as np
from IPython.display import Audio

from src.objects import MelSGram, Tune


class TuneDataset():
    """Audio Dataset pertaining short tunes.
    """
    def __init__(self,):
        pass

    def load_tunes(self,
                   folder: Path,
                   file_type: str = '*.wav'):
        """Load tunes from a given folder. The audio samples must pertain the same file type.

        Parameters
        ----------
        folder : Path
            Folder to load tunes from.
        file_type : str, optional
            File type to glob, by default '*.wav'

        Raises
        ------
        FileNotFoundError
            If folder does not exist.
        """
        self.folder = folder
        if not folder.exists():
            raise FileNotFoundError(f'Folder does not exist.')
        else:
            self.file_paths = [file for file in folder.glob(file_type)]
            for file in self.file_paths:
                assert file.exists() and file.is_file()
            return self
        
    def __repr__(self) -> str:
        return f'{len(self.file_paths)} files loaded from {self.folder}.'
    
    def __getitem__(self,
                    file_name: str):
        """Play a previously loaded file.

        Parameters
        ----------
        file_name : str
            The file name. Must contain extension.

        Returns
        -------
        Audio
            Playable audio. Suitable for Jupyter.
        """
        for file in self.file_paths:
            if file.name == file_name:
                return Audio(str(file)) 
    
    def extract_tune(self, 
                     verbose: bool =True):
        """Extract contents from loaded files.

        Parameters
        ----------
        verbose : bool, optional
            Show progress, by default True
        """
        self.tunes = list()
        for it, file_path in enumerate(self.file_paths):
            samples, sample_rate = librosa.load(str(file_path),
                                                sr = None)
            tune = Tune(
                samples = samples,
                sample_rate = sample_rate,
                file_path = file_path
            )
            self.tunes.append(tune)
            if verbose:
                print(f'{100*(it/len(self.file_paths))} %')

    def pad_tunes(self, target_time: float) -> None:
        """Pad the tunes to a target time by repeating the existing samples. Tunes with longer times are left unchanged.

        Parameters
        ----------
        target_time : float
            Target duration of the padded audio in seconds.
        """
        for tune in self.tunes:
            tune.pad(target_time)
    
    def extract_mel_sgrams(self,
                           verbose: bool = True):
        """Extract Mel Spectrogram from previously extracted file contents.

        Parameters
        ----------
        verbose : bool, optional
            Show progress, by default True
        """
        self.mel_sgrams = list()
        for it, tune in enumerate(self.tunes):
            sgram = librosa.stft(tune.samples)
            sgram_mag, _ = librosa.magphase(sgram)
            mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, 
                                                             sr=tune.sample_rate)
            # use the decibel scale to get the final Mel Spectrogram
            mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, 
                                                ref=np.min)
            mel_sg = MelSGram(
                file_path = tune.file_path,
                content = mel_sgram,
                sample_rate = tune.sample_rate
            )
            self.mel_sgrams.append(mel_sg)
            if verbose:
                print(f'{100*(it/len(self.tunes))} %')