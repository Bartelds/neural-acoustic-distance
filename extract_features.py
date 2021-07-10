from typing import Optional

KNOWN_MODELS = {
    # Pre-trained
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-large": "facebook/wav2vec2-large",
    "wav2vec2-large-lv60": "facebook/wav2vec2-large-lv60",
    "wav2vec2-large-xlsr-53": "facebook/wav2vec2-large-xlsr-53",
    # Fine-tuned
    "wav2vec2-base-960h": "facebook/wav2vec2-base-960h",
    "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
    "wav2vec2-large-960h-lv60": "facebook/wav2vec2-large-960h-lv60",
    "wav2vec2-large-960h-lv60-self": "facebook/wav2vec2-large-960h-lv60-self",
    "wav2vec2-large-xlsr-53-english": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    # Voxpopuli
    "wav2vec2-base-10k-voxpopuli": "facebook/wav2vec2-base-10k-voxpopuli",
    "wav2vec2-base-100k-voxpopuli": "facebook/wav2vec2-base-100k-voxpopuli",
    "wav2vec2-large-100k-voxpopuli": "facebook/wav2vec2-large-100k-voxpopuli",
    "wav2vec2-base-10k-voxpopuli-ft-en": "facebook/wav2vec2-base-10k-voxpopuli-ft-en",
    # Dutch
    "wav2vec2-large-xlsr-53-dutch": "facebook/wav2vec2-large-xlsr-53-dutch",
    "wav2vec2-large-xlsr-53-dutch-wietsedv": "wietsedv/wav2vec2-large-xlsr-53-dutch",
    "wav2vec2-large-nl-voxpopuli": "facebook/wav2vec2-large-nl-voxpopuli",
    "wav2vec2-base-10k-voxpopuli-ft-nl": "facebook/wav2vec2-base-10k-voxpopuli-ft-nl",
}


def load_wav2vec2_featurizer(model: str, layer: Optional[int] = None):
    """
    Loads Wav2Vec2 featurization pipeline and returns it as a function.

    Featurizer returns a list with all hidden layer representations if "layer" argument is None.
    Otherwise, only returns the specified layer representations.
    """
    from transformers.models.wav2vec2 import Wav2Vec2Model
    import soundfile as sf
    from scipy import signal
    import torch
    import numpy as np

    model_name_or_path = KNOWN_MODELS.get(model, model)
    model_kwargs = {}
    if layer is not None:
        model_kwargs["num_hidden_layers"] = layer if layer > 0 else 0
    model = Wav2Vec2Model.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    @torch.no_grad()
    def _featurize(path):
        input_values, rate = sf.read(path, dtype=np.float32)
        if len(input_values.shape) == 2:
            input_values = input_values.mean(1)
        if rate != 16_000:
            new_length = int(input_values.shape[0] / rate * 16_000)
            input_values = signal.resample(input_values, new_length)

        input_values = torch.from_numpy(input_values).unsqueeze(0)
        if torch.cuda.is_available():
            input_values = input_values.cuda()

        if layer is None:
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
            hidden_states = [s.squeeze(0).cpu().numpy() for s in hidden_states]
            return hidden_states

        if layer >= 0:
            hidden_state = model(input_values).last_hidden_state.squeeze(0).cpu().numpy()
        else:
            hidden_state = model.feature_extractor(input_values)
            hidden_state = hidden_state.transpose(1, 2)
            if layer == -1:
                hidden_state = model.feature_projection(hidden_state)
            hidden_state = hidden_state.squeeze(0).cpu().numpy()

        return hidden_state

    return _featurize


def main():
    import os
    from pathlib import Path
    from argparse import ArgumentParser

    import numpy as np
    from tqdm import tqdm

    parser = ArgumentParser(
        prog="Wav2Vec2 Featurizer",
        description="Runs full featurization of wav files for downstream usage.",
    )
    parser.add_argument("-i", "--input_dir", default="wav", type=Path)
    parser.add_argument("-o", "--output_dir", default="feats/{model}/{speaker}")
    parser.add_argument("-n", "--output_name", default="layer-{layer}.npy")
    parser.add_argument("-m", "--model", default="wav2vec2-base")
    parser.add_argument("-l", "--layer", default=None, type=int)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    # Check wav files in input directory
    wav_paths = list(args.input_dir.glob("*.wav"))
    if len(wav_paths) == 0:
        print(f"No wav files found in {args.input_dir}")
        exit(1)
    print(f"Featurizing {len(wav_paths):,} wav files")

    # Load acoustic model
    featurizer = load_wav2vec2_featurizer(args.model, layer=args.layer)

    # Create features for each wav file
    for wav_path in tqdm(wav_paths, ncols=80):
        # Check output directory
        speaker = os.path.splitext(wav_path.name)[0]
        output_dir = Path(args.output_dir.format(model=args.model, speaker=speaker))
        if not args.force and output_dir.exists() and os.listdir(output_dir):
            tqdm.write(f"{output_dir} already exists and it is not empty. skipping")
            continue

        # Extract features
        hidden_states = featurizer(wav_path)
        if args.layer is not None:
            hidden_states = [hidden_states]

        # Save features
        os.makedirs(output_dir, exist_ok=True)
        for layer, hidden_state in enumerate(hidden_states, start=args.layer or 0):
            feat_path = output_dir / args.output_name.format(layer=str(layer).zfill(2))
            np.save(feat_path, hidden_state)

        tqdm.write(str(output_dir))

    print("Done!")


if __name__ == "__main__":
    main()
