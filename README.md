<div align="center">

<h2><a href="https://github.com/bytedance/tarsier">Tarsier: Recipes for Training and Evaluating Large Video Description Models</a></h2>

Jiawei Wang*, Liping Yuan*, Yuchen Zhang*

ByteDance Research

*:Equal contribution, sorted alphabetically.
</div>

<!-- [![Paper](https://img.shields.io/badge/cs.CV-2311.17005-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2311.17005) -->

[![arXiv](https://img.shields.io/badge/arXiv-2404.16994-b31b1b.svg)](https://arxiv.org/abs/2407.00634)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/omni-research/Tarsier-34b)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg)](https://huggingface.co/datasets/omni-research/DREAM-1K)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/video-question-answering-on-mvbench)](https://paperswithcode.com/sota/video-question-answering-on-mvbench?p=tarsier-recipes-for-training-and-evaluating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/zero-shot-video-question-answer-on-next-qa)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-next-qa?p=tarsier-recipes-for-training-and-evaluating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/zero-shot-video-question-answer-on-egoschema-1)](https://paperswithcode.com/sota/zero-shot-video-question-answer-on-egoschema-1?p=tarsier-recipes-for-training-and-evaluating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/zeroshot-video-question-answer-on-msvd-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msvd-qa?p=tarsier-recipes-for-training-and-evaluating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/zeroshot-video-question-answer-on-tgif-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-tgif-qa?p=tarsier-recipes-for-training-and-evaluating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/zeroshot-video-question-answer-on-activitynet)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-activitynet?p=tarsier-recipes-for-training-and-evaluating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tarsier-recipes-for-training-and-evaluating/zeroshot-video-question-answer-on-msrvtt-qa)](https://paperswithcode.com/sota/zeroshot-video-question-answer-on-msrvtt-qa?p=tarsier-recipes-for-training-and-evaluating)


<div align="center">
  <a href="https://github.com/bytedance/tarsier">
    <img src="assets/figures/tarsier_logo.jpg" width = "60%">
  </a>
</div>

# Perface
Welcome to Tarsier!

In this repository, we introduce Tarsier -- a family of large-scale video-language models, which is designed to generate high-quality video descriptions (see Figure 1), together with good capability of general video understanding (SOTA results on 6 open benchmarks). Tarsier takes a simple model structure (CLIP-ViT + LLM), combined with a carefully designed training strategy: multi-task pre-training (stage-1) and multi-grained instruction tuning (stage-2).

Besides the model, we propose a new video desription benchmark called DREAM-1K (<b>D</b>escription
with <b>R</b>ich <b>E</b>vents, <b>A</b>ctions, and <b>M</b>otions), featuring videos from diverse sources and varying complexity. AutoDQ (<b>Auto</b>matic <b>D</b>escription Quality) is also introduced as a highly interpretable and discriminative approach to evaluate video description quality.

We have released the model, code, and data for inference, evaluation and depolyment.

- Model:

  | Model      | Link                                                                |
  | -----------|------------------------------------------------------------------------------------------------------------- |
  | Tarsier-7b  | https://huggingface.co/omni-research/Tarsier-7b |
  | Tarsier-34b | https://huggingface.co/omni-research/Tarsier-34b |

- Code: https://github.com/bytedance/tarsier

- Dataset: https://huggingface.co/datasets/omni-research/DREAM-1K

Please <a href="#citeus">cite us</a> if you found our work helpful. 
<div align="center">
  <img src="assets/figures/chatbot-example.png" width = "100%">
  <br>Figure 1: Example dialogue between a user and Tarsier. The input video is: <a href="https://github.com/bytedance/tarsier/assets/videos/coffee.gif">assets/videos/coffee.gif</a>
</div>

# Overview

### Abstract
<!-- <details> -->
Generating fine-grained video descriptions is a fundamental challenge in video understanding. In this work, we introduce Tarsier, a family of large-scale video-language models designed to generate high-quality video descriptions. Tarsier employs CLIP-ViT to encode frames separately and then uses an LLM to model temporal relationships. Despite its simple architecture, we demonstrate that with a meticulously designed two-stage training procedure, the Tarsier models exhibit substantially stronger video description capabilities than any existing open-source model, showing a +51.4% advantage in human side-by-side evaluation over the strongest model. Additionally, they are comparable to state-of-the-art proprietary models, with a +12.3% advantage
against GPT-4V and a −6.7% disadvantage against Gemini 1.5 Pro. Besides video description, Tarsier proves to be a versatile generalist model, achieving new state-of-the-art results across nine public benchmarks, including multi-choice VQA, open-ended VQA, and zero-shot video captioning. Our second contribution is the introduction of a new benchmark for evaluating video description models, consisting of a new challenging dataset featuring videos from diverse sources and varying complexity, along with an automatic method specifically designed to assess the quality of fine-grained video descriptions. We make our models and evaluation benchmark publicly available at https://github.com/bytedance/tarsier.
<!-- </details> -->

### Simple Model Structure
Tarsier takes a simple sturcture that use a MLP projection layer to connect visual encoder (CLIP ViT) and text decoder (LLM). Frames are encoded independently and concatenated to input into LLM.
<div align="center">
  <img src="assets/figures/model-arch.png" width = "90%">
  <br>Figure 2: Tarsier Model Structure.
</div>

### Two-stage Training
Tarsier tasks a two-stage training strategy.
1. Stage-1: Multi-task Pre-training
  
    In stage-1, we trained our model across:
    - 10M diverse public datasets, such as video captioning, video question answering, action recognition, multi-image understanding, and text generation.
    - 3.5M in-house data, including 2.4M high-quality video caption data similar to WebVid and 1.1M videos with object-tracking (processed on videos from Webvid and HD-VILA by object tracking tool: [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA))
2. Stage-2: Multi-grained Instruction Tuning

    In stage-2, we use 500K of in-house instruction tuning data, including:
    - Movie clips featuring multiple shots, subjects, or events, and had annotators provide descriptions varying in length and detail, from brief motion summaries to comprehensive narratives of visual details.
    - A dataset rich in camera motions, including zooming, translating, panning, and rotating.
    - Video-aware creative writing, such as poems, dialogues, speeches.
  
In both stages, we freeze ViT and train all the parameters of projection layer and LLM.

### Video Description Evaluation
#### Benchmark: DREAM-1K
We proposed DREAM-1K as a challenging video description benchmark. It contrains a collection of 1,000 video clips with diverse complexities from five different origins: live-action movies, animated movies, stock videos, long YouTube videos, and TikTok-style short videos. We provide a fine-grained manual annotation for each video. See: [data/annotations/DREAM-1k.jsonl](https://github.com/bytedance/tarsier/data/annotations/DREAM-1k.jsonl)
<div align="center">
  <img src="assets/figures/dream-1k-statistics.png" width = "90%">
  <br>Figure 3: DREAM-1K data Statistics.
</div>

#### Evaluation Approach: AutoDQ
We propose AutoDQ as a more interpretable approach to automatic video description evaluation. AutoDQ uses an extraction model to extract events from two video descriptions, then uses an entailment model to examine how many events extracted from one description are entailed by the other description. We use ChatGPT to implement both models, as shown in Figure 4.
<div align="center">
  <img src="assets/figures/automatic-evaluation.png" width = "90%">
  <br>Figure 4: The AutoDQ workflow.</a>
</div>

The relative code is: [evaluation/metrics/evaluate_dream_gpt.py](https://github.com/bytedance/tarsier/evaluation/metrics/evaluate_dream_gpt.py)

#### Evaluation Results
We evaluate some advanced open-source video understanding models and two proprietary models (GPT-4V and Genmini 1.5 Pro) on DREAM-1K. The results are shown in Figure 5.
<div align="center">
  <img src="assets/figures/dream_1k_results.png" width = "100%">
  <br>Figure 5: Evaluation results on DREAM-1K.
</div>

### Video Understanding Benchmarks Evaluation
Tarsier is evluated on 7 commonly used video understanding benchmarks, including MVBench, NeXT-QA, Egoschema, MSVD-QA, MSR-VTT-QA, ActivityNet-QA and TGIF-QA. Ours Tarsier-34b gains 6 SOTA results among the 7 benchmarks.

# Usage
This section provides guidance on how to run, evaluate and deploy Tarsier.
## Setup
Following all are running under the environment of python 3.9. If you are not using python 3.9, you can create a virtual environment with:
```
conda create -n tarsier python=3.9
```
Then run the setup script:
```bash
git clone https://github.com/bytedance/tarsier.git
cd tarsier

bash setup.sh
```
Note that you should fill in the environment parameters for calling OpenAI Service through Azure, if you need to run evaluations based on ChatGPT (for Open-ended QA and DREAM-1K).

## Model Prepare
Download the model checkpoints from Hugging Face: [Tarsier-7b](https://huggingface.co/omni-research/Tarsier-7b) and [Tarsier-34b](https://huggingface.co/omni-research/Tarsier-34b).

## Quick Start
You can use the following script to run a quick start of video detailed description:

```bash
MODEL_NAME_OR_PATH="path_to_the_model"
VIDEO_FILE="assets/videos/coffee.gif" # Or try your own example, could be images (include gif images), videos.

python3 -m tasks.inference_quick_start \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --instruction "Describe the video in detail." \
  --input_path $VIDEO_FILE
```
The result should be:
```bash
# Tarsier-7b:
"Prediction: A person is sitting at an outdoor café table with a cup of coffee and a book. The person takes a sip from the cup and then looks at the book. The background features the Eiffel Tower and is surrounded by red roses. Another person walks by in the background, waving at the first person. The scene is set in a Parisian outdoor café with street lamps and greenery visible."

# Tarsier-34b:
"Prediction: A person picks up a cup of coffee from a table and takes a sip while holding a book. The scene is set in an outdoor café with the Eiffel Tower visible in the background. The person then turns to look at two other individuals who are seated at a nearby table. The two individuals wave at the person with the coffee. The person continues to hold the coffee cup and book, looking back at the two individuals."
```


## Benchmark Inference and Evaluation
### Data Prepare
1. DREAM-1K
  
    Download Video from https://huggingface.co/datasets/omni-research/DREAM-1K.

    We have preprocessed the metadata for all benchmarks we used, see: [data/annotations](https://github.com/bytedance/tarsier/data/annotations) But you need to change the _"\<placeholder\>"_ in the annotation file to your local video file path according to the _"vid"_. We provide an [example code](https://github.com/bytedance/tarsier/code/Tarsier/data/fill_in_video_file.ipynb) for processing DREAM-1K. You can refer to it when processing other benchmarks.

2. Other Benchmarks
    - Multi-choice VQA: [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [NeXT-QA](https://github.com/doc-doc/NExT-QA) and [Egoschema](https://drive.google.com/drive/folders/1SS0VVz8rML1e5gWq7D7VtP1oxE2UtmhQ)
    - Open-ended VQA: [MSVD-QA](https://opendatalab.com/OpenDataLab/MSVD), [MSR-VTT-QA](https://opendatalab.com/OpenDataLab/MSR-VTT), [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) and [TGIF-QA](https://opendatalab.com/OpenDataLab/TGIF-QA)
    - Video Caption: [MSVD-Caption](https://opendatalab.com/OpenDataLab/MSVD), [MSRVTT-Caption](https://opendatalab.com/OpenDataLab/MSR-VTT), [VATEX](https://eric-xw.github.io/vatex-website/about.html)

### Benchmark Inference and Evaluation
Following command will firstly run in parallel to inference on the selected benchmarks (Edit the parameters in [scripts/run_inference_benchmark.sh](https://github.com/bytedance/tarsier/scripts/run_inference_benchmark.sh): _"CHUNKS"_ and _"GPULIST"_ to customly control the parallelism), and then run evaluation.
```bash
model_name_or_path="path_to_the_model"
output_dir="dream_predictions"
benchmarks="dream" # Split benchmarks by space. Default as 'all' to inference on all benchmarks; Also could be task types: ('dream', 'caption', 'mc_qa', 'oe_qa'); Or specific benchmark names: ('dream', 'msvd-caption', 'msr-vtt-caption', 'vatex-caption', 'next-qa', 'egoschema', 'mvbench', 'video-mme', 'msvd-qa', 'msr-vtt-qa', 'tgif-qa', 'anet-qa')

mkdir $output_dir

bash scripts/run_inference_benchmark.sh $model_name_or_path $output_dir $benchmarks
```
The evaluation results will be printed and saved in _$output_dir_.

### Evaluation Only
Run the following script to only calcluate the metrics for selected benchmarks.
```bash
pred_dir="dream_predictions"
benchmarks="dream" # Same as above code block

bash run_evaluation_only.sh $pred_dir $benchmark
```
The evaluation result will be save as: _{pred_dir}/{benchmark-name}\_eval\_result.txt_

## Deployment
### CLI Demo
Use the following script to run a conversation demo in command line.
```bash
model_path="path_to_the_model"

bash scripts/run_demo_cli.sh $model_path
```
Bellow is the input video and a conversation with Tarsier-34b about the video:
<div align="center">
  <img src="assets/videos/demo_test.gif" width = "100%">
  <br>Figure 6: Input video in CLI Demo.</a>
</div>
<br>
<div align="center">
  <img src="assets/videos/demo_cli_example.gif" width = "100%">
  <br>Figure 7: Conversation in CLI Demo.</a>
</div>

### Gradio Demo
Use the following script to run a Gradio Demo.
```bash
model_path="path_to_the_model"

bash scripts/run_demo_gradio.sh $model_path
```

The gradio page show be as following. You shoud input a Video/Image/GIF in according block firstly, and then start conversation. Click the __"Clear"__ button to restart.

<div align="center">
  <img src="assets/figures/gradio_page.png" width = "100%">
  <br>Figure 8: Tarsier Gradio Demo.</a>
</div>

# <span id="citeus">Citation</span>
Pleae cite us as:

```BibTeX
@misc{wang2024tarsierrecipestrainingevaluating,
      title={Tarsier: Recipes for Training and Evaluating Large Video Description Models}, 
      author={Jiawei Wang and Liping Yuan and Yuchen Zhang},
      year={2024},
      eprint={2407.00634},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.00634},
}
```
