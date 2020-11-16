import glob
import pandas as pd
from tqdm import tqdm
from speech_reps.featurize import DeCoARFeaturizer
import torch 
import soundfile as sf
from python_speech_features import mfcc
import pickle

folderpath = r'./data/saa/'
accentpath = './resources/us_ratings.csv'

wav_files = sorted(glob.glob(folderpath + "/wav/*.wav"))
align_files = sorted(glob.glob(folderpath + "/txg/*.txt"))
accent_data = pd.read_csv(accentpath, sep=" ")[['id', 'Nativelikeness']]

featurizer = DeCoARFeaturizer('./artifacts/decoar-encoder-29b8e2ac.params')

speaker_to_features = dict()

def f(start, end, tensor):
    return tensor[int(start):int(end), :]

for i in tqdm(range(len(align_files))):
    align_file, wav_file = align_files[i], wav_files[i]
    speaker = align_files[i].split('/')[-1][:-4]
    if speaker == 'sinhalese4':
        speaker = 'sinhala4'
    
    alignments = pd.read_csv(align_file, sep="\t", header=None).iloc[:, 1:]
    alignments.columns = ["Word", "Start", "End"]
    alignments = alignments[alignments["Word"] != 'sp'].reset_index(drop = True)
    alignments[["Start", "End"]] = alignments[["Start", "End"]].apply(lambda x: round(x * 100))

    native_likeness = None

    if speaker in set(accent_data['id']):
        native_likeness = accent_data[accent_data['id'] == speaker]['Nativelikeness'].values[0]

    features = torch.tensor(featurizer.file_to_feats(wav_file)).view(-1, 2048).detach().numpy()
    
    alignments["Features"] = alignments.apply(lambda x: f(x.Start, x.End, features), axis = 1)
    speaker_to_features[speaker] = (alignments, native_likeness)


with open('decoar_saa_us.pickle', 'wb') as handle:
    pickle.dump(speaker_to_features, handle)
