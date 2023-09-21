# Isolating Child Speech using Convolutional Neural Networks

## Replicating Experiments

### Setup

`!git clone https://github.com/davidwmcdevitt/pedzstar-speech-pipeline`
`!pip install -r requirements.txt`

### Preparing Data Directories

Experiments are initialized from the raw audio files of the Speech Exemplars and Evaluation Database (SEED). Access to the SEED dataset can be requested at (https://pedzstarlab.soc.northwestern.edu/) 

###

## Background

Gathering targeted child speech data in both research and classroom settings consequently involves the capture of non-targeted adult speech in the recordings, often as prompts for the child speaker or in conversational turn-taking tasks [^1^]. To utilize this data in future work, researchers must undergo a labor-intensive identification process to isolate the child speech, thereby constraining resources and limiting the growth of extensive child speaker datasets for speech pathology research.

Deep learning toolkits have made rapid advancements in audio and speech applications possible. While Open Source speaker diarization pipelines like OpenAI's Whisper [^2^] address the "who said what and when" problem, they struggle to differentiate between child and adult speakers. This limitation arises from various intrinsic and extrinsic factors affecting child speech.

Our proposal offers an alternative approach using a convolutional neural network (CNN) architecture that better addresses the needs of isolating child speech.

## Methodology

We employ deep learning methods such as CNNs, which have proven effective in audio classification tasks. Our CNN architecture comprises six convolutional blocks, each containing a 2D convolutional layer. The model can classify audio segments as belonging to child or adult speakers and can also handle more complex scenarios, such as overlapping speech between a child and an adult.

For this study, we use the Speech Exemplars and Evaluation Database (SEED), which contains 17,000 utterances from 69 child and 33 adult speakers collected in clinical and classroom settings [^5^].

## Citations

[^1^]: [M. Speights et al., "Automated episode selection of child continuous speech via blind source extraction," in J. Acoust. Soc. Am.](https://doi.org/10.1121/1.5068583)  
[^2^]: [A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision,"](https://doi.org/10.48550/arXiv.2212.04356)  
[^3^]: [L. Nanni et al., "An Ensemble of Convolutional Neural Networks for Audio Classification," Applied Sciences](https://doi.org/10.48550/arXiv.2007.07966)  
[^4^]: [K. J. Piczak, "Environmental sound classification with convolutional neural networks," 2015 IEEE 25th International Workshop on Machine Learning for Signal Processing](https://ieeexplore.ieee.org/document/7324337)  
[^5^]: [Speights Atkins et al., Speech exemplar and evaluation database (SEED) for clinical training in articulatory phonetics and speech science](https://osf.io/ygc8n/?view_only=e5a044f04c8a435aaa808efbfd3297e6)  
