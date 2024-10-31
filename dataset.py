import json
import os
import pathlib

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

from transformers import ClapModel, ClapProcessor


# matplotlib.use('TkAgg')
# Get the color map by name:
cm = plt.get_cmap('jet')

FS = 48000
MIN_DURATION = 0.3
MAX_DURATION = 10

NFFT = 2048
WIN_LEN = 2048
HOP_LEN = 124
WIN_OVERLAP = WIN_LEN - HOP_LEN


class DetectionsDataset:
    def __init__(self, config):
        # Folders
        self.wavs_files_df = config['wavs_files_df']
        self.wavs_folder = config['wavs_folder']
        self.spectrograms_folder = pathlib.Path(config['spectrograms_folder'])
        self.snippets_folder = pathlib.Path(config['snippets_folder'])
        self.annotations_file = pathlib.Path(config['annotations_file'])

        self.channel = 0
        self.config = config

    def __setitem__(self, key, value):
        if key in self.config.keys():
            self.config[key] = value
        self.__dict__[key] = value

    def save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def create_snippet_spectrogram(self, snippet):
        f, t, sxx = scipy.signal.spectrogram(snippet, fs=FS, window=('hamming'),
                                             nperseg=WIN_LEN,
                                             noverlap=WIN_OVERLAP, nfft=NFFT,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        sxx = sxx[f > 50, :]
        sxx = 10 * np.log10(sxx)
        per_min = np.percentile(sxx.flatten(), 10)
        per_max = np.percentile(sxx.flatten(), 99)
        sxx = (sxx - per_min) / (per_max - per_min)
        sxx[sxx < 0] = 0
        sxx[sxx > 1] = 1
        #sxx = cm(sxx)  # convert to color

        sxx = 1 - sxx
        img = np.array(sxx[:, :] * 255, dtype=np.uint8)
        return img, f

    def all_snippets(self, detected_foregrounds):

        file_list = os.listdir(self.wavs_folder)
        for i, row in tqdm(detected_foregrounds.iterrows(), total=len(detected_foregrounds)):
            wav_path = row['wav']
            waveform_info = torchaudio.info(wav_path)

            # If the selection is in between two files, open both and concatenate them
            if row['begin_sample'] > row['end_sample']:
                waveform1, fs = torchaudio.load(wav_path,
                                                frame_offset=row['begin_sample'],
                                                num_frames=waveform_info.num_frames - row[
                                                    'begin_sample'])

                wav_path2 = self.wavs_folder.joinpath(file_list[file_list.index(row['wav_name']) + 1])
                waveform2, fs = torchaudio.load(wav_path2,
                                                frame_offset=0,
                                                num_frames=row['end_sample'])
                waveform = torch.cat([waveform1, waveform2], -1)
            else:
                waveform, fs = torchaudio.load(wav_path,
                                               frame_offset=row['begin_sample'],
                                               num_frames=row['end_sample'] - row[
                                                   'begin_sample'])
            if waveform_info.sample_rate > FS:
                waveform = F.resample(waveform=waveform, orig_freq=fs, new_freq=FS)[self.channel, :]
            else:
                waveform = waveform[self.channel, :]

            waveform = 0.8 * 2 * (waveform - waveform.min()) / (waveform.max()  - waveform.min()) - 1
            yield i, row, waveform

    def load_relevant_selection_table(self):
        annotations_file = pathlib.Path(self.annotations_file)
        selections = pd.read_table(annotations_file)

        # Filter the selections
        selections = selections.loc[selections['Low Freq (Hz)'] < (FS / 2)]
        selections = selections.loc[selections.View == 'Spectrogram 1']
        selections.loc[selections['High Freq (Hz)'] > (FS / 2), 'High Freq (Hz)'] = FS / 2
        selections = selections.loc[
            (selections['End Time (s)'] - selections['Begin Time (s)']) >= MIN_DURATION]
        selections = selections.loc[
            (selections['End Time (s)'] - selections['Begin Time (s)']) <= MAX_DURATION]

        selections = selections.rename(columns={'High Freq (Hz)': 'max_freq',
                                                'Low Freq (Hz)': 'min_freq',
                                                'Begin File': 'wav_name',
                                                'Beg File Samp (samples)': 'begin_sample',
                                                'End File Samp (samples)': 'end_sample',
                                                'Tags': 'class',
                                                'Source': 'source'
                                                 })

        selections = selections[['min_freq', 'max_freq', 'wav_name', 'begin_sample', 'end_sample', 'class']]
        selections2 = pd.merge(selections, self.wavs_files_df, left_on='wav_name', right_on='wav_name')
        return selections2

    def process(self, max_duration=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ClapModel.from_pretrained("davidrrobinson/BioLingual",
                                          cache_dir=r"C:\Users\cleap\Documents\Data\models\huggingface").to(device)
        processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", sampling_rate=FS,
                                                  cache_dir=r"C:\Users\cleap\Documents\Data\models\huggingface")

        selections = self.load_relevant_selection_table()
        features_list, idxs = [], []
        for i, row, waveform in tqdm(self.all_snippets(selections), total=len(selections)):
            # Create snippet and spectrohram
            snippet_name = str(row.deployment_id) + '_' + str(row.name) + '.wav'
            spectrogram_name = snippet_name.replace('.wav', '.png')
            sf.write(self.snippets_folder.joinpath(snippet_name), waveform, samplerate=FS)
            img, f = self.create_snippet_spectrogram(waveform)
            Image.fromarray(np.flipud(img)).save(self.spectrograms_folder.joinpath(spectrogram_name))

            # compute embeddings
            x = [waveform.cpu().numpy()]
            inputs = processor(audios=x, return_tensors="pt", sampling_rate=FS).to(device)
            audio_embed = model.get_audio_features(**inputs)
            features_list.extend(audio_embed.cpu().detach().numpy())
            idxs.append(i)

        features_space = torch.Tensor(np.stack(features_list).astype(float))
        features_df = pd.DataFrame(features_space.numpy())
        features_df.index = idxs
        df = pd.merge(features_df, selections, left_index=True, right_index=True)
        return df
