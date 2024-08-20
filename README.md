# Finetunable-VAD

프로젝트의 디렉터리 구조는 다음과 같습니다.

```plaintext
Finetunable-VAD
├── configs
├── app
├── old_data
│   ├── Audio
│   │   ├── Female
│   │   │   ├── PTDB-TUG
│   │   │   └── TMIT
│   │   ├── Male
│   │   │   ├── PTDB-TUG
│   │   │   └── TMIT
│   │   └── Noizeus
│   │       ├── Babble
│   │       ├── Car
│   │       ├── NoNoise
│   │       └── ...
│   ├── Audio_1
│   │   ├── Female
│   │   │   ├── PTDB-TUG
│   │   │   └── TMIT
│   │   ├── Male
│   │   │   ├── PTDB-TUG
│   │   │   └── TMIT
│   │   └── Noizeus
│   │       ├── Babble
│   │       ├── Car
│   │       ├── NoNoise
│   │       └── ...
│   ├── google_dataset_v2
│   │   └── google_speech_recognition_v2
│   │       ├── backward
│   │       ├── bed
│   │       ├── bird
│   │       └── ...
│   └── ...
├── src
│   ├── create_data
│   ├── data_preprocessing
│   ├── fine_tune
│   └── vad_algorithm
└── ...
