from argparse import ArgumentParser
import os

import torch
import torch.onnx
import soundfile as sf
import numpy as np

from extract_features import KNOWN_MODELS

from transformers.models.wav2vec2 import Wav2Vec2Model

parser = ArgumentParser()
parser.add_argument("-m", "--model", default="wav2vec2-base")
parser.add_argument("-l", "--layer", default=9, type=int)
parser.add_argument("-e", "--example_path", default="wav/english1.wav")
parser.add_argument("-o", "--output_path", default="onnx/{model}-layer-{layer}.onnx")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

layer_str = str(args.layer).zfill(2)

input_values, rate = sf.read(args.example_path, dtype=np.float32)
assert rate == 16_000
input_values = torch.from_numpy(input_values).unsqueeze(0)

model = Wav2Vec2Model.from_pretrained(KNOWN_MODELS.get(args.model, args.model), num_hidden_layers=args.layer)
torch.onnx.export(model,
                  input_values,
                  args.output_path.format(model=args.model, layer=layer_str),
                  input_names=["input_values"],
                  dynamic_axes={
                      "input_values": [1],
                  })
