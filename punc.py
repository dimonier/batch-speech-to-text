import torch
import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification

# Прогнозируемые знаки препинания
PUNK_MAPPING = {'.': 'PERIOD', ',': 'COMMA', '?': 'QUESTION'}
PUNCTUATION = ''.join(PUNK_MAPPING.keys())

# Прогнозируемый регистр LOWER - нижний регистр, UPPER - верхний регистр для первого символа,
# UPPER_TOTAL - верхний регистр для всех символов
LABELS_CASE = ['LOWER', 'UPPER', 'UPPER_TOTAL']
# Добавим в пунктуацию метку O означающий отсутсвие пунктуации
LABELS_PUNC = ['O'] + list(PUNK_MAPPING.values())

# Сформируем метки на основе комбинаций регистра и пунктуации
LABELS_list = []
for case in LABELS_CASE:
    for punc in LABELS_PUNC:
        LABELS_list.append(f'{case}_{punc}')
LABELS = {label: i+1 for i, label in enumerate(LABELS_list)}
LABELS['O'] = -100
INVERSE_LABELS = {i: label for label, i in LABELS.items()}

LABEL_TO_PUNC_LABEL = {label: label.split('_')[-1] for label in LABELS.keys() if label != 'O'} # type: ignore
LABEL_TO_CASE_LABEL = {label: '_'.join(label.split('_')[:-1]) for label in LABELS.keys() if label != 'O'} # type: ignore

MODEL_REPO = "kontur-ai/sbert_punc_case_ru"

def token_to_label(token, label):
    if type(label) == int:
        label = INVERSE_LABELS[label]
    if label == 'LOWER_O':
        return token
    if label == 'LOWER_PERIOD':
        return token + '.'
    if label == 'LOWER_COMMA':
        return token + ','
    if label == 'LOWER_QUESTION':
        return token + '?'
    if label == 'UPPER_O':
        return token.capitalize()
    if label == 'UPPER_PERIOD':
        return token.capitalize() + '.'
    if label == 'UPPER_COMMA':
        return token.capitalize() + ','
    if label == 'UPPER_QUESTION':
        return token.capitalize() + '?'
    if label == 'UPPER_TOTAL_O':
        return token.upper()
    if label == 'UPPER_TOTAL_PERIOD':
        return token.upper() + '.'
    if label == 'UPPER_TOTAL_COMMA':
        return token.upper() + ','
    if label == 'UPPER_TOTAL_QUESTION':
        return token.upper() + '?'
    if label == 'O':
        return token

def decode_label(label, classes='all'):
    if classes == 'punc':
        return LABEL_TO_PUNC_LABEL[INVERSE_LABELS[label]]
    if classes == 'case':
        return LABEL_TO_CASE_LABEL[INVERSE_LABELS[label]]
    else:
        return INVERSE_LABELS[label]

class SbertPuncCase(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO,
                                                       strip_accents=False, cache_dir = r'D:\Python\_models\transformer')
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_REPO, cache_dir = r'D:\Python\_models\transformer')
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask)

    def punctuate(self, text):
        text = text.strip().lower()

        # Разобъем предложение на слова
        words = text.split()

        tokenizer_output = self.tokenizer(words, is_split_into_words=True)

        if len(tokenizer_output.input_ids) > 512:
            return ' '.join([self.punctuate(' '.join(text_part)) for text_part in np.array_split(words, 2)])

        predictions = self(torch.tensor([tokenizer_output.input_ids], device=self.model.device),
                           torch.tensor([tokenizer_output.attention_mask], device=self.model.device)).logits.cpu().data.numpy()
        predictions = np.argmax(predictions, axis=2)

        # decode punctuation and casing
        splitted_text = []
        word_ids = tokenizer_output.word_ids()
        for i, word in enumerate(words):
            label_pos = word_ids.index(i)
            label_id = predictions[0][label_pos]
            label = decode_label(label_id)
            splitted_text.append(token_to_label(word, label))
        capitalized_text = ' '.join(splitted_text)
        return capitalized_text
