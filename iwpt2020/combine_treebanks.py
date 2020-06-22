# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-09 07:45

import glob
import os

from iwpt2020 import cdroot


def load_conll(path):
    with open(path) as src:
        text = src.read()
        sents = text.split('\n\n')
        sents = [x for x in sents if x.strip()]
        return sents


def combine(folders, file, out):
    sents = []
    for f in folders:
        f = glob.glob(f'{f}/*{file}')
        if not f:
            continue
        assert len(f) == 1
        f = f[0]
        sents += load_conll(f)
    with open(out, 'w') as out:
        out.write('\n\n'.join(sents))
        out.write('\n\n')


def main():
    print('Combing treebanks of the same language ...')
    cdroot()
    ISO639_1 = {
        'ab': 'Abkhaz',
        'aa': 'Afar',
        'af': 'Afrikaans',
        'ak': 'Akan',
        'sq': 'Albanian',
        'am': 'Amharic',
        'ar': 'Arabic',
        'an': 'Aragonese',
        'hy': 'Armenian',
        'as': 'Assamese',
        'av': 'Avaric',
        'ae': 'Avestan',
        'ay': 'Aymara',
        'az': 'Azerbaijani',
        'bm': 'Bambara',
        'ba': 'Bashkir',
        'eu': 'Basque',
        'be': 'Belarusian',
        'bn': 'Bengali',
        'bh': 'Bihari',
        'bi': 'Bislama',
        'bs': 'Bosnian',
        'br': 'Breton',
        'bg': 'Bulgarian',
        'my': 'Burmese',
        'ca': 'Catalan; Valencian',
        'ch': 'Chamorro',
        'ce': 'Chechen',
        'ny': 'Chichewa; Chewa; Nyanja',
        'zh': 'Chinese',
        'cv': 'Chuvash',
        'kw': 'Cornish',
        'co': 'Corsican',
        'cr': 'Cree',
        'hr': 'Croatian',
        'cs': 'Czech',
        'da': 'Danish',
        'dv': 'Divehi; Maldivian;',
        'nl': 'Dutch',
        'dz': 'Dzongkha',
        'en': 'English',
        'eo': 'Esperanto',
        'et': 'Estonian',
        'ee': 'Ewe',
        'fo': 'Faroese',
        'fj': 'Fijian',
        'fi': 'Finnish',
        'fr': 'French',
        'ff': 'Fula',
        'gl': 'Galician',
        'ka': 'Georgian',
        'de': 'German',
        'el': 'Greek, Modern',
        'gn': 'Guaraní',
        'gu': 'Gujarati',
        'ht': 'Haitian',
        'ha': 'Hausa',
        'he': 'Hebrew (modern)',
        'hz': 'Herero',
        'hi': 'Hindi',
        'ho': 'Hiri Motu',
        'hu': 'Hungarian',
        'ia': 'Interlingua',
        'id': 'Indonesian',
        'ie': 'Interlingue',
        'ga': 'Irish',
        'ig': 'Igbo',
        'ik': 'Inupiaq',
        'io': 'Ido',
        'is': 'Icelandic',
        'it': 'Italian',
        'iu': 'Inuktitut',
        'ja': 'Japanese',
        'jv': 'Javanese',
        'kl': 'Kalaallisut',
        'kn': 'Kannada',
        'kr': 'Kanuri',
        'ks': 'Kashmiri',
        'kk': 'Kazakh',
        'km': 'Khmer',
        'ki': 'Kikuyu, Gikuyu',
        'rw': 'Kinyarwanda',
        'ky': 'Kirghiz, Kyrgyz',
        'kv': 'Komi',
        'kg': 'Kongo',
        'ko': 'Korean',
        'ku': 'Kurdish',
        'kj': 'Kwanyama, Kuanyama',
        'la': 'Latin',
        'lb': 'Luxembourgish',
        'lg': 'Luganda',
        'li': 'Limburgish',
        'ln': 'Lingala',
        'lo': 'Lao',
        'lt': 'Lithuanian',
        'lu': 'Luba-Katanga',
        'lv': 'Latvian',
        'gv': 'Manx',
        'mk': 'Macedonian',
        'mg': 'Malagasy',
        'ms': 'Malay',
        'ml': 'Malayalam',
        'mt': 'Maltese',
        'mi': 'Māori',
        'mr': 'Marathi (Marāṭhī)',
        'mh': 'Marshallese',
        'mn': 'Mongolian',
        'na': 'Nauru',
        'nv': 'Navajo, Navaho',
        'nb': 'Norwegian Bokmål',
        'nd': 'North Ndebele',
        'ne': 'Nepali',
        'ng': 'Ndonga',
        'nn': 'Norwegian Nynorsk',
        'no': 'Norwegian',
        'ii': 'Nuosu',
        'nr': 'South Ndebele',
        'oc': 'Occitan',
        'oj': 'Ojibwe, Ojibwa',
        'cu': 'Old Church Slavonic',
        'om': 'Oromo',
        'or': 'Oriya',
        'os': 'Ossetian, Ossetic',
        'pa': 'Panjabi, Punjabi',
        'pi': 'Pāli',
        'fa': 'Persian',
        'pl': 'Polish',
        'ps': 'Pashto, Pushto',
        'pt': 'Portuguese',
        'qu': 'Quechua',
        'rm': 'Romansh',
        'rn': 'Kirundi',
        'ro': 'Romanian, Moldavan',
        'ru': 'Russian',
        'sa': 'Sanskrit (Saṁskṛta)',
        'sc': 'Sardinian',
        'sd': 'Sindhi',
        'se': 'Northern Sami',
        'sm': 'Samoan',
        'sg': 'Sango',
        'sr': 'Serbian',
        'gd': 'Scottish Gaelic',
        'sn': 'Shona',
        'si': 'Sinhala, Sinhalese',
        'sk': 'Slovak',
        'sl': 'Slovene',
        'so': 'Somali',
        'st': 'Southern Sotho',
        'es': 'Spanish; Castilian',
        'su': 'Sundanese',
        'sw': 'Swahili',
        'ss': 'Swati',
        'sv': 'Swedish',
        'ta': 'Tamil',
        'te': 'Telugu',
        'tg': 'Tajik',
        'th': 'Thai',
        'ti': 'Tigrinya',
        'bo': 'Tibetan',
        'tk': 'Turkmen',
        'tl': 'Tagalog',
        'tn': 'Tswana',
        'to': 'Tonga',
        'tr': 'Turkish',
        'ts': 'Tsonga',
        'tt': 'Tatar',
        'tw': 'Twi',
        'ty': 'Tahitian',
        'ug': 'Uighur, Uyghur',
        'uk': 'Ukrainian',
        'ur': 'Urdu',
        'uz': 'Uzbek',
        've': 'Venda',
        'vi': 'Vietnamese',
        'vo': 'Volapük',
        'wa': 'Walloon',
        'cy': 'Welsh',
        'wo': 'Wolof',
        'fy': 'Western Frisian',
        'xh': 'Xhosa',
        'yi': 'Yiddish',
        'yo': 'Yoruba',
        'za': 'Zhuang, Chuang',
        'zu': 'Zulu',
    }
    testfiles = sorted(glob.glob('data/iwpt2020/test-blind/*.txt'))
    trainfiles = sorted(glob.glob('data/iwpt2020/train-dev/*'))
    trainfiles = [x for x in trainfiles if glob.glob(x + '/*.conllu')]
    print(trainfiles)
    for idx, txt in enumerate(testfiles):
        basename = os.path.basename(txt)
        langcode = basename.split('.')[0]
        lang = ISO639_1.get(langcode)
        fs = [x for x in trainfiles if lang.lower() in x.lower()]
        print(f'{idx + 1:02d}/{len(testfiles)} {basename} {[os.path.basename(x) for x in fs]}')
        folder = f'data/iwpt2020/train-dev-combined/{langcode}'
        os.makedirs(folder, exist_ok=True)
        for part in ['train', 'dev']:
            combine(fs, f'{part}.enhanced_collapse_empty_nodes.conllu', f'{folder}/{part}.conllu')


if __name__ == '__main__':
    main()
