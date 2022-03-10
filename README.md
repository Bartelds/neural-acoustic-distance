# Neural Representations for Modeling Variation in Speech
Code associated with the paper: Neural Representations for Modeling Variation in Speech.

## Citation

```bibtex
@article{bartelds2021neural,
  title = {Neural representations for modeling variation in speech},
  author = {Martijn Bartelds and Wietse {de Vries} and Faraz Sanal and Caitlin Richter and Mark Liberman and Martijn Wieling},
  journal = {Journal of Phonetics},
  volume = {92},
  pages = {101137},
  year = {2022},
  issn = {0095-4470},
  doi = {https://doi.org/10.1016/j.wocn.2022.101137},
  url = {https://www.sciencedirect.com/science/article/pii/S0095447022000122}
}
```

## Installation

```bash
git clone https://github.com/Bartelds/neural-acoustic-distance.git
cd neural-acoustic-distance
pip install -r requirements.txt
```

## Required files
 - `/wav`
   - `{speaker_name}.wav`: recording of a speaker.
 - `/timestamps`
   - `{speaker_name}.txt`: tab seperated list with word, start time and end time.
 - `/lists`
   - `{name}.txt`: tab seperated list with speakers and optionally perceptual ratings.

## Data

All Speech Accent archive data is available [online](https://accent.gmu.edu/).

The recordings used in this study are listed in `wav/wav-ids.txt`.

The other datasets used for this study are available on request.

## Usage

### PyTorch

```bash
python measure_distance.py -m wav2vec2-base -l 9
```

Works with any `w2v2` or `XLSR` model available on the [Huggingface](https://huggingface.co/models?search=wav2vec2) 🤗 .

Enter the corresponding model name after the model (`-m`) argument, and choose the output layer using the layer argument (`-l`).

To get faster distance measures, you can pre-compute the features:

```bash
python extract_features.py -m wav2vec2-base -l 9
python measure_distance.py -m wav2vec2-base -l 9 -a "feats/{model}/{speaker}/layer-{layer}.npy"
```

### ONNX
To use the ONNX runtime instead of Transformers + PyTorch, convert the model to ONNX first:

```bash
python convert_to_onnx.py -m wav2vec2-base -l 9
python measure_distance.py -m wav2vec2-base -l 9 --onnx
```

## Benchmarks
These results show the total run time for two `w2v2` models using a specified output layer on the Speech Accent Archive dataset.

Hardware:
 - CPU: Intel Core i7-6900K
 - GPU: Nvidia Titan Xp

### w2v2-base layer 9: PyTorch (GPU)
```
$ time python measure_distance.py -m wav2vec2-base -l 9

Correlation: -0.83

real    4m54.508s
user    12m55.299s
sys     2m22.281s
```

### w2v2-base layer 9: ONNX (CPU)
```
$ time python measure_distance.py -m wav2vec2-base -l 9 --onnx

Correlation: -0.83

real    8m42.248s
user    50m13.038s
sys     1m43.175s
```

### w2v2-large layer 10: PyTorch (GPU)
```
$ time python measure_distance.py -m wav2vec2-large -l 10

Correlation: -0.85

real    6m42.438s
user    15m38.194s
sys     3m9.954s
```

### w2v2-large layer 10: ONNX (CPU)
```
$ time python measure_distance.py -m wav2vec2-large -l 10 --onnx

Correlation: -0.85

real    11m51.411s
user    70m52.092s
sys     2m15.958s
```

## Visualization tool
We introduce a visualization tool that allows us to better understand the neural models. 

These visualizations show the contribution of the neural features to the acoustic distances by highlighting where the distance between pronunciations is highest.

The [online demo](https://colab.research.google.com/drive/193xTirkkgwzK9pDeYBPiQ0ehnVPJrtNn) is publicly available, such that visualizations can be created of your own data using one of the Transformer-based neural methods evaluated in our paper.

## Optional: non Transformer-based models
Required files for using the non Transformer-based models are in the `fairseq` folder.

To use DeCoAR embeddings you also need to install [Kaldi](https://github.com/kaldi-asr/kaldi) first.

## Required files
 - `/data`
   - `/wav`
   - - `{speaker_name}.wav`: recording of a speaker.
   - `/txg`
   - - `{speaker_name}.txt`: tab seperated list with word, start time, and end time.
 - `/resources`
   - - `{name}.csv`: list with speakers and optionally perceptual ratings.
   - - `{tgt_name}.txt`: list with target speakers.
   - - `{ref_name}.txt`: list with reference speakers.

### Downloading pre-trained models

This part works with the pre-trained models released by [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md) and [awslabs](https://github.com/awslabs/speech-representations).

### Create DeCoAR features

To use `DeCoAR` features in the pipeline, run the following command.

```bash
python fairseq/scripts/create_decoar.py
```

## Run the model

 - `method`: One of `w2v`, `w2v-a`, `w2v-q`, `w2v-qa`, `vqw2v`, or `decoar`.
 - `model`: One of the pre-trained model released by [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md).
 - `dataset`: Set to `us` for Speech Accent Archive dataset (default) or `dsd`, `nos`, `nl` for any of the other datasets.
 - `layer`: The index of the hidden layer to be used in the Transformer models.
 - `no-dtw`: If only features need to be calculated, dynamic time warping can be disabled.

The following command will compute acoustic distances and the correlation with human perception using `vqw2v` layer 9 on the Speech Accent Archive dataset.

```bash
python fairseq/scripts/process_embeddings.py \
        --method vqw2v \
        --model vq-wav2vec_kmeans \
        --dataset us \
        --layer 9
```

## Reproducibility

Running the code of this repository might result in minimally different results compared to those reported in the paper, due to version updates of the [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md), [awslabs](https://github.com/awslabs/speech-representations) or [Huggingface](https://huggingface.co/models?search=wav2vec2) models.
