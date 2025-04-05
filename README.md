## Batch Audio and Video to Text Converter with Punctuation and Case Recovery for Russian Language

**Input**: audio/video file(s)

**Output**: text + text with timecodes (optional)

## Description

This script uses OpenAI's Whisper model or Hugging Face models for speech-to-text conversion and SBert model for punctuation and case recovery in Russian language. Supports the following formats: mp3, aac, wav, ogg, m4a, mp4, opus, mov, avi, mkv, webm.

GPU (discrete graphics card) is recommended, but CPU is also supported (slower).

## Limitations

- CUDA driver is required for GPU operation
- Recognition quality depends on the chosen model (larger models provide better results but work slower)
- Punctuation and case recovery works only for Russian language

## Setup

1. Install [Python](https://python.org) 3.10+

2. Install PyTorch
- with CUDA support (if you plan to use GPU). For example, for CUDA 12.6:
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
For installation details of other versions, check here: https://pytorch.org/

   To check which CUDA version is installed on your computer, in Windows open console (Win+R, cmd) and run `nvidia-smi`. The CUDA version will be shown at the beginning of the report.

- without CUDA support (if your computer doesn't have a discrete graphics card or CUDA is not required): `pip install torch`

3. Install [Whisper](https://github.com/openai/whisper) and transformers libraries: `pip install openai-whisper transformers`

4. Install compiled [FFMPEG](https://ffmpeg.org/download.html) and add the path to `ffmpeg.exe` to the `path` environment variable

## Usage

1. Run the script. If all dependencies are installed, the models will be downloaded from the Internet (only on first run) and recognition of the specified file or folder will begin.
2. The recognition progress is displayed during program execution.
3. Upon completion, two text files will appear next to each source file:
   - filename.txt - recognized text, one sentence per line.
   - filename_timecode.txt - recognized text with timecodes, approximately 3-5 seconds per line.

### Command Line Parameters

- `-i`, `--input`: source media file name for text conversion
- `-f`, `--folder`: folder with media files for processing (current folder by default)
- `-d`, `--device`: device for model execution (`cpu` or `cuda`, default is `cpu`)
- `-m`, `--model`: Whisper model name or Hugging Face model path (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, or any Hugging Face model path, default is `large-v3-turbo`)
- `-l`, `--language`: force recognition in specified language (e.g., `ru`). If not set, language is detected automatically
- `-r`, `--raw`: disable punctuation and case correction (enabled by default)
- `-t`, `--timecode`: enable extra output file containing timecodes and text (disabled by default)

### Simple launch with default settings

`python batch-speech-to-text.py -i record.mp3`

### Advanced launch with parameters

`python batch-speech-to-text.py -f d:\ASR -m medium -d cuda -l ru -t`

where:
- `-f d:\ASR` - folder with media files for processing
- `-m medium` - use Whisper `medium` model
- `-d cuda` - use GPU for faster recognition
- `-l ru` - force Russian language for recognition
- `-t` - enable timecode generation

### Using Hugging Face models

You can use any Hugging Face model by providing its path:

`python batch-speech-to-text.py -i record.mp3 -m dvislobokov/whisper-large-v3-turbo-russian`

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [SBert Punctuation and Case Model](https://huggingface.co/kontur-ai/sbert_punc_case_ru)