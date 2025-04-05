import argparse
import os
import re
import tempfile
import subprocess
import time
from datetime import timedelta

import whisper
from transformers import pipeline

# Use script's folder if the folder not specified
current_folder = os.path.dirname(os.path.abspath(__file__))

# Audio files with specified extensions will be processed
audio_exts = ['mp3', 'aac', 'wav', 'ogg', 'm4a', 'mp4', 'opus', 'mov']
video_exts = ['mp4', 'mov', 'avi', 'mkv', 'webm']

parser = argparse.ArgumentParser("Batch speech-to-text converter with punctuation and case restoration")
parser.add_argument("-i", "--input", type=str, help="input media file to convert to text", default='')
parser.add_argument("-f", "--folder", type=str, help="folder contaiing files to process", default=current_folder)
parser.add_argument("-d", "--device", type=str, help="run model on cpu or gpu", choices=['cpu', 'cuda'], default='cpu')
parser.add_argument("-m", "--model", type=str, help="whisper model name or huggingface model path", default='large-v3-turbo')
parser.add_argument("-l", "--language", type=str, help="force recognition using this language, e.g. ru. Detected automatically if not specified")
parser.add_argument("-r", "--raw", action="store_true", help="do not fix punctuation and case", default=False)
parser.add_argument("-t", "--timecode", action="store_true", help="create a file with timecodes", default=False)

args = parser.parse_args()
audio_folder = args.folder
model_name = args.model
text_language = args.language
device = args.device

# Determine if the model is from Hugging Face by checking if the name contains '/'
use_huggingface = '/' in model_name

if not args.raw:
    from punc import *
    print(f"Loading punctuation and case model", end="... ")
    try:
        sbertpunc = SbertPuncCase().to("cpu")
        print(f"done")
    except Exception as e:
        print(f"failed: {e}")

def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video file using FFmpeg and save it as a temporary WAV file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path to the temporary audio file
    """
    print(f"Extracting audio from video file...", end="")
    try:
        # Create a temporary file for the audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()
        
        # Use FFmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file
            temp_audio.name
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print("done")
        return temp_audio.name
    except subprocess.CalledProcessError as e:
        print(f"failed: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"failed: {e}")
        raise

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds using FFmpeg.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse FFmpeg output to find duration
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                time_str = line.split('Duration:')[1].split(',')[0].strip()
                h, m, s = time_str.split(':')
                duration = int(h) * 3600 + int(m) * 60 + float(s)
                return duration
        return 0
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

def main():
    lang_text = f"to {text_language} " if text_language else ""
    print(f"Starting speech-to-text recognition {lang_text}on {device} device")
    
    if use_huggingface:
        print(f"Loading huggingface model: {model_name}", end="... ")
        try:
            model = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device
            )
            print(f"done")
        except Exception as e:
            print(f"failed: {e}")
            return
    else:
        print(f"Loading whisper model: {model_name}", end="... ")
        try:
            model = whisper.load_model(model_name).to(device)
            print(f"done")
        except Exception as e:
            print(f"failed: {e}")
            return

# Processing a single file
    if args.input:
        print(f'\nProcessing file: {args.input}')
        process_audiofile(args.input, model)
        return

# Processing folder
    print(f'\nProcessing folder: {audio_folder}')
    os.chdir(audio_folder)

    files = [file for file in os.listdir(audio_folder) if match_ext(file, audio_exts)]
    print(f'Found {len(files)} files.')

    for filename in files:
        print(f'\nProcessing file: {filename}')
        audio_file = os.path.join(audio_folder, filename)
        process_audiofile(audio_file, model)


def match_ext(filename: str, extensions):
    return filename.lower().split('.')[-1] in extensions


def process_audiofile(fname: str, model):
    fext = fname.split('.')[-1].lower()
    fname_noext = fname[:-(len(fext)+1)]
    
    # If it's a video file, extract audio first
    if fext in video_exts:
        try:
            audio_file = extract_audio_from_video(fname)
            should_cleanup = True
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return
    else:
        audio_file = fname
        should_cleanup = False

    try:
        # Get audio duration
        audio_duration = get_audio_duration(audio_file)
        print(f"Audio duration: {timedelta(seconds=int(audio_duration))}")

        # Start timing
        start_time = time.time()

        if use_huggingface:
            print("Processing audio with Hugging Face model...")
            result = model(
                audio_file,
                return_timestamps=True,
            )
            segments = []
            for chunk in result["chunks"]:
                if chunk["timestamp"][0] is not None:  # Проверяем наличие временной метки
                    segments.append({
                        "start": chunk["timestamp"][0],
                        "text": chunk["text"]
                    })
                else:
                    segments.append({
                        "start": 0,  # Если временная метка отсутствует, используем 0
                        "text": chunk["text"]
                    })
        else:
            result = model.transcribe(audio_file, verbose=False, language=text_language)
            segments = result['segments']

        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Processing time: {timedelta(seconds=int(processing_time))}")
        if audio_duration > 0:
            speed = audio_duration / processing_time
            print(f"Processing speed: {speed:.2f}x real-time")

        if args.timecode:
            with open(fname_noext + '_timecode.txt', 'w', encoding='UTF-8') as f:
                for segment in segments:
                    timecode_sec = int(segment['start'])
                    hh = timecode_sec // 3600
                    mm = (timecode_sec % 3600) // 60
                    ss = timecode_sec % 60
                    timecode = f'[{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}]'
                    text = segment['text']
                    f.write(f'{timecode} {text}\n')

        rawtext = ' '.join([segment['text'].strip() for segment in segments])
        rawtext = re.sub(" +", " ", rawtext)

        alltext = re.sub("([\.\!\?]) ", "\\1\n", rawtext)

        if not args.raw:
            alltext = fix_punctuation(alltext)

        with open(fname_noext + '.txt', 'w', encoding='UTF-8') as f:
            f.write(alltext)
    finally:
        # Clean up temporary audio file if it was created
        if should_cleanup and os.path.exists(audio_file):
            os.unlink(audio_file)

def fix_punctuation(text: str) -> str:
    text = text.splitlines()

    target_text = ''
    chunks = 0
    for sentence in text:
        # print(f'Len: {len(sentence)}')
        contains_punc = False
        for sign in PUNCTUATION:
            if sign in sentence:
                contains_punc = True
        if (len(sentence) > 80 and contains_punc == False) or len(sentence) > 300:
            chunks += 1
            # print(f"Source text:   {sentence}\n")
            punctuated_text = sbertpunc.punctuate(sentence.strip(PUNCTUATION))
            punctuated_text_nodoubles = re.sub(",([,\.\!\?])", "\\1", punctuated_text)
            # print(f"Restored text: {punctuated_text_nodoubles}")
            punctuated_text_lines = re.sub("([\.\!\?]) ", "\\1\n", punctuated_text_nodoubles)
            target_text += punctuated_text_lines + '\n'
        else:
            target_text += sentence + '\n'
    print(f"Fixed punctuation and case in {chunks} chunks")
    return target_text

if __name__ == '__main__':

    # Calling main() function
    main()
