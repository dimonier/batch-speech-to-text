import argparse
import os
import re

import whisper

# Use script's folder if the folder not specified
current_folder = os.path.dirname(os.path.abspath(__file__))

# Audio files with specified extensions will be processed
audio_exts = ['mp3', 'aac', 'wav', 'ogg', 'm4a', 'mp4', 'opus', 'mov']

parser = argparse.ArgumentParser("Batch speech-to-text converter with punctuation and case restoration")
parser.add_argument("-i", "--input", type=str, help="input media file to convert to text", default='')
parser.add_argument("-f", "--folder", type=str, help="folder contaiing files to process", default=current_folder)
parser.add_argument("-d", "--device", type=str, help="run model on cpu or gpu", choices=['cpu', 'cuda'], default='cpu')
parser.add_argument("-m", "--model", type=str, help="whisper model for speech-to-text", choices=['tiny', 'base', 'small', 'medium', 'large-v3', 'large-v3-turbo'], default='large-v3-turbo')
parser.add_argument("-l", "--language", type=str, help="force recognition using this language, e.g. ru. Detected automatically if not specified")
parser.add_argument("-r", "--raw", action="store_true", help="do not fix punctuation and case", default=False)

args = parser.parse_args()
audio_folder = args.folder
whisper_model = args.model
text_language = args.language
device = args.device

if not args.raw:
    from punc import *
    print(f"Loading punctuation and case model", end="... ")
    try:
        sbertpunc = SbertPuncCase().to("cpu")
        print(f"done")
    except Exception as e:
        print(f"failed: {e}")

def main():
    lang_text = f"to {text_language} " if text_language else ""
    print(f"Starting speech-to-text recognition {lang_text}on {device} device")
    print(f"Loading whisper model: {whisper_model}", end="... ")
    try:
        model = whisper.load_model(whisper_model).to(device)
#        model = model.to(device)
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
    return filename.split('.')[-1] in extensions


def process_audiofile(fname: str, model: whisper.model.Whisper):

    fext = fname.split('.')[-1]
    fname_noext = fname[:-(len(fext)+1)]

    result = model.transcribe(fname, verbose = False, language = text_language)

    with open(fname_noext + '_timecode.txt', 'w', encoding='UTF-8') as f:
        for segment in result['segments']:
            timecode_sec = int(segment['start'])
            hh = timecode_sec // 3600
            mm = (timecode_sec % 3600) // 60
            ss = timecode_sec % 60
            timecode = f'[{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}]'
            text = segment['text']
        #                print(f'{timecode} {text}\n')
            f.write(f'{timecode} {text}\n')

    rawtext = ' '.join([segment['text'].strip() for segment in result['segments']])
    rawtext = re.sub(" +", " ", rawtext)

    alltext = re.sub("([\.\!\?]) ", "\\1\n", rawtext)

    if not args.raw:
        alltext = fix_punctuation(alltext)

    with open(fname_noext + '.txt', 'w', encoding='UTF-8') as f:
        f.write(alltext)

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
