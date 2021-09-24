# Voice Activity Detection

## 1. Installation

```
$ conda create -n vad python=3.9
$ conda activate vad
$ git clone https://github.com/curly1/voice-activity-detection.git
$ cd voice-activity-detection/
$ pip install -r requirements.txt
$ pip install -e .
```

## 2. Scope

The goal of the project was to develop a Voice Activity Detection system using Neural Networks. The overall approach is as follows:

- [Lhotse library](https://github.com/lhotse-speech/lhotse) was used to prepare the data;
- PyTorch was used for the NN modelling part;
- 