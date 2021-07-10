from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from dtw import dtw
from scipy.stats import pearsonr
from tqdm import tqdm

_cache = {}


def read_speaker_list(path, extra_col=None):
    import pandas as pd

    df = pd.read_csv(path, sep=" ")
    assert isinstance(df, pd.DataFrame)
    names = df["name"].tolist()
    if extra_col is not None:
        return names, df[extra_col].tolist()
    return names


def load_features(time_path,
                  feat_path,
                  rate,
                  featurizer_fn,
                  blacklist=[],
                  whitelist=[]) -> List[Tuple[str, np.ndarray]]:
    blacklist = ['sp'] if len(blacklist) == 0 else ['sp'] + blacklist
    whitelist = None if len(whitelist) == 0 else whitelist

    if str(feat_path).endswith(".npy"):
        feats = np.load(feat_path)
    else:
        if "featurizer" not in _cache:
            _cache["featurizer"] = featurizer_fn()
        feats = _cache["featurizer"](feat_path)

    token_feats = []
    with open(time_path) as f:
        for line in f:
            _, token, start, end = line.rstrip().split("\t")
            if whitelist and token not in whitelist:
                continue
            if token in blacklist:
                continue
            start = round(float(start) * rate)
            end = round(float(end) * rate)
            token_feats.append((token, feats[start:end]))
    return token_feats


def load_onnx_wav2vec2_featurizer(onnx_path):
    import onnxruntime as ort
    import soundfile as sf

    print("ORT Device:", ort.get_device())

    ort_session = ort.InferenceSession(onnx_path)

    def _featurize(path):
        input_values, rate = sf.read(path, dtype=np.float32)
        assert rate == 16_000
        input_values = input_values.reshape(1, -1)

        hidden_state = ort_session.run(None, {"input_values": input_values})[0].squeeze(0)
        return hidden_state

    return _featurize


def _dtw_distance(a_token, a_feats, b_token, b_feats):
    assert a_token == b_token

    if a_feats.shape[0] < 2 or b_feats.shape[0] < 2:
        return np.nan

    return dtw(
        a_feats,
        b_feats,
        distance_only=True,
    ).normalizedDistance


def pairwise_dtw_distance(A, B, pool=None):
    """
    Computes average distance between aligned pairs in two lists of sequences.

    Uses a pool for parallel processing
    """
    if pool is None:
        dists = [_dtw_distance(*a, *b) for a, b in zip(A, B)]
    else:
        dists = pool.starmap(_dtw_distance, [(*a, *b) for a, b in zip(A, B)])

    return np.nanmean(dists)


def main():
    parser = ArgumentParser()
    # Input Data:
    parser.add_argument("-a", "--input_feat_path", default="wav/{speaker}.wav")
    parser.add_argument("-i", "--input_time_path", default="timestamps/{speaker}.txt")
    parser.add_argument("-r", "--reference_list", default="lists/us_ref.txt", type=Path)
    parser.add_argument("-t", "--target_list", default="lists/us_tgt.txt", type=Path)
    parser.add_argument("--target_column", default="Nativelikeness")
    parser.add_argument("--blacklist", default=["A"], nargs="*")
    parser.add_argument("--whitelist", default=[], nargs="*")
    # Model:
    parser.add_argument("-m", "--model", default="wav2vec2-base")
    parser.add_argument("-l", "--layer", default=9, type=int)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--onnx_path", default="onnx/{model}-layer-{layer}.onnx")
    parser.add_argument("--rate", default=50, type=int)
    # Misc:
    parser.add_argument("-p", "--num_procs", default=4, type=int)
    args = parser.parse_args()
    print(args)

    if args.onnx and args.input_feat_path.endswith(".npy"):
        print(f"ONNX cannot be used with cached features")
        exit(1)

    layer_str = str(args.layer).zfill(2)

    def featurizer_fn():
        if args.onnx:
            return load_onnx_wav2vec2_featurizer(args.onnx_path.format(model=args.model, layer=layer_str))
        from extract_features import load_wav2vec2_featurizer
        return load_wav2vec2_featurizer(args.model, args.layer)

    # Load target and reference lists
    tgt_speakers, tgt_scores = read_speaker_list(args.target_list, args.target_column)
    ref_speakers = read_speaker_list(args.reference_list)

    # Determine input/output paths
    input_feat_path = args.input_feat_path.format(
        model=args.model,
        layer=layer_str,
        speaker="{speaker}",
    )

    def load_features_fn(speaker):
        return load_features(
            args.input_time_path.format(speaker=speaker),
            input_feat_path.format(speaker=speaker),
            featurizer_fn=featurizer_fn,
            blacklist=args.blacklist,
            whitelist=args.whitelist,
            rate=args.rate,
        )

    # Preload reference features
    refs_tokens_features = [load_features_fn(speaker) for speaker in ref_speakers]

    # Calculate distances
    with Pool(args.num_procs) as pool:
        tgt_distances = []
        for tgt_i, tgt_speaker in enumerate(tqdm(tgt_speakers, ncols=80), start=1):
            tgt_tokens_features = load_features_fn(tgt_speaker)
            dists = []
            for ref_tokens_features in refs_tokens_features:
                d = pairwise_dtw_distance(ref_tokens_features, tgt_tokens_features, pool=pool)
                dists.append(d)
            tgt_distances.append(np.mean(dists))

            if tgt_i % 5 == 0:
                r, _ = pearsonr(tgt_distances, tgt_scores[:tgt_i])
                tqdm.write(f"{tgt_i:>4}/{len(tgt_speakers)}: {r*100:.1f}")

    r, _ = pearsonr(tgt_distances, tgt_scores)
    print(f"\nCorrelation: {r*100:.2f}")


if __name__ == "__main__":
    main()
