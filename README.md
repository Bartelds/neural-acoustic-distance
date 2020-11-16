# Neural Representations for Modeling Variation in English Speech
Code associated with the paper: Neural Representations for Modeling Variation in English Speech.


<!-- ## Citation


```bibtex

``` -->

## Installation

```bash
git clone https://github.com/Bartelds/neural-acoustic-distance.git
cd neural-acoustic-distance
pip install -r requirements.txt
```

To use DeCoAR embeddings you also need to install [Kaldi](https://github.com/kaldi-asr/kaldi).

### Downloading pre-trained models

This repository works with the pre-trained models released by [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md) and [awslabs](https://github.com/awslabs/speech-representations). 


### Downloading data

All Speech Accent archive data is available [online](https://accent.gmu.edu/). Check `data/wav-ids.txt` for the correct files.
The Dutch speakers dataset used for this study is available on request.

### Create DeCoAR features

To use DeCoAR features in the pipeline, run the following command.

```bash
python create_decoar.py
```

## Run the model

 - `method`: One of `w2v`, `w2v-a`, `w2v-q`, `w2v-qa`, `vqw2v`, `w2v2`, `w2v2-q`, `w2v2-qa`, or `decoar`. See paper for explaination of the model suffixes.
 - `model`: One of the pre-trained model released by [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md).
 - `dataset`: Set to `us` for Speech Accent Archive dataset (default) or `dsd` for Dutch speakers dataset.
 - `layer`: The index of the hidden layer to be used in the Transformer models.
 - `no-dtw`: If only features need to be calculated, dynamic time warping can be disabled.

The following command will compute acoustic distances and the correlation with human perception using wav2vec 2.0 (Large) with Transformer layer 10 on the Speech Accent Archive dataset.

```bash
python process_embeddings.py \
        --method w2v2-qa \
        --model libri960_big \
        --dataset us \
        --layer 10
```
