import os
import torch
import pickle
import glob
import math
import soundfile as sf
import numpy as np
import pandas as pd
import argparse

from dtw import dtw
from tqdm import tqdm
from scipy.stats import pearsonr

from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel


class FilesDataset:
    def __init__(self, folderpath, accentpath, method, suffix):
        self.method = method

        self.wav_files = sorted(glob.glob(folderpath + f"/wav{suffix}/*.wav"))
        self.align_files = sorted(glob.glob(folderpath +
                                            f"/txg{suffix}/*.txt"))
        self.accent_data = pd.read_csv(accentpath,
                                       sep=" ")[['id', 'Nativelikeness']]

        if len(self.wav_files) == 0:
            print('no wav files in', folderpath + f"/wav{suffix}/*.wav")
            exit(1)

        assert len(self.wav_files) == len(self.align_files)

    def __len__(self):
        return len(self.align_files)

    def __getitem__(self, index):

        align_file = self.align_files[index]
        wav_file = self.wav_files[index]

        assert align_file.split('/')[-1][:-4] == wav_file.split('/')[-1][:-4]
        speaker = align_file.split('/')[-1][:-4]
        if speaker == 'sinhalese4':
            speaker = 'sinhala4'

        wav, sr = sf.read(wav_file)
        assert sr == 16000

        wav = torch.from_numpy(wav).unsqueeze(0).float()

        alignments = pd.read_csv(align_file, sep="\t", header=None).iloc[:, 1:]
        alignments.columns = ["Word", "Start", "End"]
        alignments = alignments[alignments["Word"] != 'sp'].reset_index(
            drop=True)
        if self.method.split('-')[0] == 'w2v2':
            alignments = alignments[alignments["Word"] != 'A'].reset_index(
                drop=True)
            alignments[["Start",
                        "End"]] = alignments[["Start", "End"
                                              ]].apply(lambda x: round(x * 50))
        else:
            alignments[["Start", "End"
                        ]] = alignments[["Start", "End"
                                         ]].apply(lambda x: round(x * 100))

        native_likeness = None
        if speaker in set(self.accent_data['id']):
            native_likeness = self.accent_data[
                self.accent_data['id'] == speaker]['Nativelikeness'].values[0]
        return speaker, alignments, wav, native_likeness


def extract_wav2vec2_hiddens(model: Wav2Vec2Model,
                             source,
                             layer_i=None,
                             cuda_limit=320_000):
    layer_i = None if layer_i is None or layer_i < 0 else layer_i - 1

    if source.shape[1] > cuda_limit:
        source = source.cpu()
        model.cpu()

    x = model.feature_extractor(source)
    x = x.transpose(1, 2)
    x = model.layer_norm(x)
    x = model.post_extract_proj(x)

    # Transformer encoder
    x_conv = model.encoder.pos_conv(x.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)
    x += x_conv

    x = model.encoder.layer_norm(x)
    x = x.transpose(0, 1)

    for i, layer in enumerate(model.encoder.layers):
        x, z = layer(x, self_attn_padding_mask=None, need_weights=False)
        if i == layer_i:
            break

    if torch.cuda.is_available():
        model.cuda()

    x = x.transpose(0, 1)
    return x


class Wav2Vec_Accent:
    def __init__(self, folderpath, accentpath, refpath, tgtpath, wav2vecpath,
                 bertpath, method, suffix):

        # Paths
        self.folderpath = folderpath
        self.accentpath = accentpath
        self.wav2vecpath = wav2vecpath
        self.bertpath = bertpath
        self.method = method
        self.suffix = suffix

        # Data
        self.data = self.load_data()
        self.references = pd.read_csv(refpath, sep=" ")['name'].tolist()
        self.targets = pd.read_csv(tgtpath, sep=" ")['name'].tolist()

        # Models
        self.model = self.load_wav2vec(wav2vecpath)
        self.roberta = self.load_bert(bertpath)
        self.features = None

    def load_data(self):
        print("LOADING DATA....", self.folderpath, self.accentpath,
              self.method, self.suffix)

        dataset = FilesDataset(self.folderpath, self.accentpath, self.method,
                               self.suffix)

        return dataset

    def load_wav2vec(self, wav2vecpath):
        if not wav2vecpath:
            return None

        print("LOADING WAV2VEC....")

        cp = torch.load(wav2vecpath, map_location=torch.device('cpu'))
        if self.method.split('-')[0] == 'w2v2':
            model = Wav2Vec2Model.build_model(cp['args'], task=None)
        else:
            model = Wav2VecModel.build_model(cp['args'], task=None)

        model.load_state_dict(cp['model'])
        model = model.eval()
        if torch.cuda.is_available():
            print('moving WAVE2VEC to CUDA')
            model.cuda()
        return model

    def load_bert(self, bertpath):
        if not bertpath:
            return None

        print("LOADING BERT....")

        roberta = RobertaModel.from_pretrained(
            bertpath, checkpoint_file='bert_kmeans.pt')
        roberta = roberta.eval()
        if torch.cuda.is_available():
            print('moving ROBERTA to CUDA')
            roberta.cuda()
        return roberta

    def extract_baseline(self, baselinepath):

        speaker_to_baseline = dict()

        filenames = glob.glob(self.folderpath + "/txg/*.txt")
        accent_data = pd.read_csv(self.accentpath,
                                  sep=" ")[['id', 'Nativelikeness']]

        plps = pd.read_csv(baselinepath, sep=" ", header=None)
        plps.columns = ['Speaker', 'Word', 'Phone', 'Vowel'
                        ] + [str(i) for i in range(4, 43)]
        plps['Features'] = plps[plps.columns[4:]].apply(list, axis=1)
        plps = plps[['Speaker', 'Word', 'Features']]

        for filename in tqdm(filenames):
            speaker = filename.split('/')[-1][:-4]
            plp = plps[plps['Speaker'] == speaker]
            plp = plp.groupby(['Word'])['Features'].apply(list).reset_index()

            native_likeness = None
            if speaker in accent_data['id']:
                native_likeness = accent_data[
                    accent_data['id'] == speaker]['Nativelikeness'].values[0]

            speaker_to_baseline[speaker] = (plp, native_likeness)

        return speaker_to_baseline

    def extract_features(self,
                         split=2040,
                         aggregate=False,
                         quantize=False,
                         save=True,
                         load=False,
                         layer_i=-1,
                         save_file='speaker_to_features.pickle'):
        if load and os.path.exists(save_file):
            print(f"LOADING FROM {save_file}")
            with open(save_file, 'rb') as handle:
                self.features = pickle.load(handle)
                return self.features

        print("\nEXTRACTING...")

        speaker_to_features = dict()

        def f(start, end, tensor):
            return tensor[int(start):int(end), :]

        def indices_to_string(idxs):
            return " ".join("-".join(map(str, a.tolist()))
                            for a in idxs.squeeze(0))

        print("LENGTH OF DATA", self.data.__len__())

        for i in tqdm(range(len(self.data))):
            speaker, alignments, wav, native_likeness = self.data[i]

            if speaker not in self.references and speaker not in self.targets:
                continue

            tqdm.write(speaker)

            if torch.cuda.is_available():
                wav = wav.cuda()

            if quantize and self.method in ['w2v-q', 'w2v-qa']:
                z = self.model.feature_extractor(wav)
                z = self.model.vector_quantizer(z)['x']
            elif quantize and self.method == 'w2v2-q':
                z, _ = self.model.quantize(wav)
                z = torch.transpose(z, 1, 2)
            elif quantize and aggregate and self.method == 'w2v2-qa':
                z = extract_wav2vec2_hiddens(self.model, wav, layer_i)
                z = torch.transpose(z, 1, 2)
            else:
                z = self.model.feature_extractor(wav)

            if aggregate and self.method != 'w2v2-qa':
                z = self.model.feature_aggregator(z)

            feats = None

            if self.roberta:

                _, idxs = self.model.vector_quantizer.forward_idx(z)
                idx_str = indices_to_string(idxs)
                tokens = list(
                    self.roberta.task.source_dictionary.encode_line(
                        idx_str, append_eos=False,
                        add_if_not_exist=False).detach().numpy())

                if not split:
                    eq_parts = 1
                    while (int(len(tokens) / eq_parts) > 2046):
                        eq_parts += 1
                    split = int(len(tokens) / eq_parts)
                else:
                    split = int(split)

                multiple = int(math.ceil(len(tokens) / split))

                vec_arr = [[0] + tokens[split * i:split * (i + 1)] + [2]
                           for i in range(multiple)]
                x_arr = [torch.LongTensor(vec).unsqueeze(0) for vec in vec_arr]

                if torch.cuda.is_available():
                    x_arr = [x.cuda() for x in x_arr]

                llfs = [
                    self.roberta.extract_features(x, return_all_hiddens=True)
                    [layer_i].squeeze(0).cpu().detach().numpy()[1:-1, :]
                    for x in x_arr
                ]

                feats = np.concatenate(tuple(llfs))

            else:
                feats = np.transpose(z.squeeze(0).cpu().detach().numpy())

            tqdm.write(f'feature shape: {feats.shape}')

            alignments["Features"] = alignments.apply(
                lambda x: f(x.Start, x.End, feats), axis=1)
            speaker_to_features[speaker] = (alignments, native_likeness)

        self.features = speaker_to_features

        if save:
            print(f"SAVING to {save_file}...")
            with open(save_file, 'wb') as handle:
                pickle.dump(speaker_to_features, handle)

        return speaker_to_features

    def calculate_dtw(self,
                      prev_features=None,
                      ongoing_pr=False,
                      save=True,
                      save_file='dtw_distances.csv'):

        print("\nCALCULATING DTW...")

        calc_features = self.features

        if prev_features:
            calc_features = prev_features

        def f(index, test_features):
            return np.mean([
                dtw(test_features,
                    calc_features[ref][0]['Features'][index],
                    window_type="slantedband",
                    window_args={
                        'window_size': 200
                    }).normalizedDistance for ref in self.references
            ])

        def collect_index(row):
            return row.name

        distances = []
        native_likeness = []
        speakers = []
        not_included = {
            "english103", "english23", "english26", "english54", "english59",
            "english76"
        }

        counter = 1
        for speaker, (df, nl) in calc_features.items():

            if speaker not in not_included and nl and speaker in self.targets:

                print("\nPROGRESS {} / {}".format(counter, 280))
                counter += 1

                print("SPEAKER ", speaker)

                df['row_index'] = df.apply(collect_index, axis=1)
                distances.append(
                    np.mean(
                        df.apply(lambda x: f(x.row_index, x.Features),
                                 axis=1)))

                speakers.append(speaker)
                native_likeness.append(nl)

                if ongoing_pr and len(distances) > 2:
                    print("PR", pearsonr(distances, native_likeness))

        corr = pearsonr(distances, native_likeness)
        print("\nFINAL CORRELATION: ", corr)

        if save:
            print(f"\nSAVING to {save_file}...")

            output = pd.DataFrame(speakers)
            output.columns = ['speaker']
            output['wv_distance'] = distances
            output['native_likeness'] = native_likeness
            output.to_csv(save_file, sep='\t')

        return corr


def main():
    parser = argparse.ArgumentParser(
        description='example: python emb_accent_dtw.py w2v wav2vec_large')
    parser.add_argument('--method',
                        choices=[
                            'w2v', 'w2v-a', 'w2v-q', 'w2v-qa', 'vqw2v', 'w2v2',
                            'w2v2-q', 'w2v2-qa', 'decoar'
                        ],
                        help='Select model [vqw2v, w2v, w2v2, decoar]')
    parser.add_argument('--model')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--no-dtw', action='store_true')
    parser.add_argument('--dataset', default='us', choices=['us', 'dsd'])
    parser.add_argument('--layer', default=-1, type=int)
    args = parser.parse_args()

    datapath = 'data'
    accentpath = f'{datapath}/resources/{args.dataset}_ratings.csv'
    refpath = f'{datapath}/resources/us_reference_speakers.txt'
    tgtpath = f'{datapath}/resources/{args.dataset}_target_speakers.txt'
    decoarpath = f'{datapath}/decoar/decoar_saa_{args.dataset}.pickle'

    suffix = '' if args.dataset == 'us' else f'_{args.dataset}'
    layer_suffix = f'_layer{args.layer}' if args.layer > -1 else ''

    outpath = f'{datapath}/output/{args.method}/{args.model}'
    featpath = f'{outpath}/features{suffix}{layer_suffix}.pickle'
    distpath = f'{outpath}/distances{suffix}{layer_suffix}.csv'
    corrpath = f'{outpath}/correlation{suffix}{layer_suffix}.txt'

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    model_type = args.method.split('-')[0]
    if model_type == 'vqw2v':
        wav2vecpath = 'data/models/w2v/vq-wav2vec_kmeans.pt'
        bertpath = 'data/models/bert_kmeans'
    elif model_type == 'decoar':
        wav2vecpath = None
        bertpath = None
    else:
        wav2vecpath = f'{datapath}/models/{model_type}/{args.model}.pt'
        bertpath = None

    if wav2vecpath is not None and not os.path.exists(wav2vecpath):
        print(f'{wav2vecpath} does not exist')
        exit(1)

    w_a = Wav2Vec_Accent(datapath, accentpath, refpath, tgtpath, wav2vecpath,
                         bertpath, args.method, suffix)
    if args.method == 'decoar':
        print(f"LOADING FROM {decoarpath}")
        with open(decoarpath, 'rb') as handle:
            w_a.features = pickle.load(handle)
    else:
        w_a.extract_features(save=True,
                             load=args.load,
                             save_file=featpath,
                             aggregate=args.method
                             in ['w2v-a', 'w2v-qa', 'w2v2-qa'],
                             quantize=args.method
                             in ['w2v-q', 'w2v-qa', 'w2v2-q', 'w2v2-qa'],
                             layer_i=args.layer)

    del w_a.model
    del w_a.roberta
    torch.cuda.empty_cache()

    if not args.no_dtw:
        corr = w_a.calculate_dtw(ongoing_pr=True, save_file=distpath)
        with open(corrpath, 'w') as f:
            f.write(f'{corr}\n')


if __name__ == "__main__":
    main()
