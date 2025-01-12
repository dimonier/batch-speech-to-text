# -*- coding: utf-8 -*-
# Credits:
#  https://github.com/openai/whisper
#  dimonier@gmail.com / https://t.me/dimonier / https://github.com/dimonier/batch-speech-to-text
# To install whisper, execute:
# pip install openai-whisper
# More info: https://github.com/openai/whisper
import os
import re
import whisper

# Audio files in this folder will be processed
audio_folder = r'C:\ASR'

# whisper_model = 'tiny' # Very fast and very inaccurate speech recognition
# whisper_model = 'base' # Fast and fourius
# whisper_model = 'small' # Worse than medium but still OK
# whisper_model = 'medium' # Good recognition results but too slow on CPU
whisper_model = 'large-v3-turbo' # Almost the same size as medium, but should give better result
# whisper_model = 'large' # Use on GPU only


text_language = 'ru' # Force usage of specified language
device = 'cuda'
#device = 'cpu'

def main():

# Audio files with specified extensions will be processed
    audio_exts = ['mp3', 'aac', 'wav', 'ogg', 'm4a', 'mp4', 'opus', 'mov']

    print(f'Looking into "{audio_folder}"')
    os.chdir(audio_folder)

    files = [file for file in os.listdir(audio_folder) if match_ext(file, audio_exts)]
    print(f'Found {len(files)} files:')
    for filename in files: print(filename)

    for filename in files:
        print(f'\n\nProcessing {filename}')
        audio_file = os.path.join(audio_folder, filename)
        process_audiofile(audio_file)


def match_ext(filename, extensions):
    return filename.split('.')[-1] in extensions


def process_audiofile(fname):

    fext = fname.split('.')[-1]
    fname_noext = fname[:-(len(fext)+1)]

    model = whisper.load_model(whisper_model)
    model = model.to(device)

    result = model.transcribe(fname, verbose = True, language = text_language)

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

    with open(fname_noext + '.txt', 'w', encoding='UTF-8') as f:
        f.write(alltext)

if __name__ == '__main__':

    # Calling main() function
    main()
