import os
import os.path

import soundfile as sf
import librosa
import numpy as np
import torch
import torch.utils.data as data



MFCC = "MFCC"
MEL = "MEL"
STFT_LOG1 = "STFT_LOG1"
STFT = "STFT"



FEATURES = STFT_LOG1

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

def fnormalize(mfcc):
    newarr =[]
    for feature in range(len(mfcc)):
        min = np.min(mfcc[feature])
        max = np.max(mfcc[feature])
        avg = np.mean(mfcc[feature])
        std = np.std(mfcc[feature])
        normed = (mfcc[feature] -avg) / std
        newarr.append(normed)
    return np.array(newarr)

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(caller, dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def make_test_dataset(caller, dir, class_to_idx):
    spects = []
    fnames = os.listdir(dir)
    for fname in fnames:
        if is_audio_file(fname):
            path = os.path.join(dir, fname)
            item = (path, None)
            spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    spect = get_features(sr, hop_length, n_fft, win_length, window, y)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


def stft_get_features(sr, hop_length, n_fft, win_length, window, y):
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    return spect


def stft_log1_get_features(sr, hop_length, n_fft, win_length, window, y):
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    return spect


def mel_get_features(sr, hop_length, n_fft, win_length, window, y):
    D = librosa.feature.melspectrogram(y, n_fft=n_fft, sr=sr, hop_length=hop_length, n_mels=160)
    spect, phase = librosa.magphase(D)
    return spect

def mfcc_get_features(sr, hop_length, n_fft, win_length, window, y):
    spect = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    return spect

feature_functions = {MFCC: mfcc_get_features,
                     MEL: mel_get_features,
                     STFT: stft_get_features,
                     STFT_LOG1: stft_log1_get_features}

def get_features(sr, hop_length, n_fft, win_length, window, y):
    return feature_functions[FEATURES](sr, hop_length, n_fft, win_length, window, y)


class GCommandLoader(data.Dataset):
    make_dataset_func =make_dataset
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=False, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = self.make_dataset_func(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (spect, target, path)

    def __len__(self):
        return len(self.spects)

class GTestCommandLoader(GCommandLoader):
    make_dataset_func = make_test_dataset
    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=False, max_len=101):
        GCommandLoader.__init__(self, root, transform=None, target_transform=None, window_size=.02,
                                window_stride=.01, window_type='hamming', normalize=False, max_len=101)


    def __getitem__(self, index):
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (spect, path)