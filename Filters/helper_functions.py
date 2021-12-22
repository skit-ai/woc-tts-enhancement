import numpy as np
import torchaudio
import wave

class Helper_Functions:
    '''
    Helper functions for processing audio files.
    '''
    def Normalize_01(self, x):
        # normalize to -1.0 -- 1.0 
        # Scipy and Wave return integers, which can be normalized and converted to floats according to the number of bits of encoding.
        if x.dtype == 'int16':
            nb_bits = 16  # -> 16-bit wav files
        elif x.dtype == 'int32':
            nb_bits = 32  # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        return (x / (max_nb_bit + 1))
    
    def Denormalize_01(self, x):
        # denormalize 
        # SoundFile, Audiolab and Torchaudio returns floats between -1 and 1 (the convention for audio signals), they can be denormalized as well
        if x.dtype == 'int16':
            nb_bits = 16  # -> 16-bit wav files
        elif x.dtype == 'int32':
            nb_bits = 32  # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        return (x * (max_nb_bit + 1))