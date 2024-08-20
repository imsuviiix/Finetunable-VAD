# Finetunable-VAD

프로젝트의 디렉터리 구조는 다음과 같습니다.

```plaintext
Finetunable-VAD
├── configs
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
│   │       ├── Restaurant
│   │       ├── Station
│   │       ├── Street
│   │       └── Train
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
│   │       ├── Restaurant
│   │       ├── Station
│   │       ├── Street
│   │       └── Train
│   ├── google_dataset_v2
│   │   └── google_speech_recognition_v2
│   │       ├── backward
│   │       ├── bed
│   │       ├── bird
│   │       ├── cat
│   │       ├── dog
│   │       ├── down
│   │       ├── eight
│   │       ├── five
│   │       ├── follow
│   │       ├── forward
│   │       ├── four
│   │       ├── go
│   │       ├── happy
│   │       ├── house
│   │       ├── learn
│   │       ├── left
│   │       ├── marvin
│   │       ├── nine
│   │       ├── no
│   │       ├── off
│   │       ├── on
│   │       ├── one
│   │       ├── right
│   │       ├── seven
│   │       ├── sheila
│   │       ├── six
│   │       ├── stop
│   │       ├── three
│   │       ├── tree
│   │       ├── two
│   │       ├── up
│   │       ├── visual
│   │       ├── wow
│   │       ├── yes
│   │       ├── zero
│   │       ├── _background_noise_
│   │       └── _background_noise_more
│   │           └── _background_noise_
│   ├── json
│   ├── noise_data
│   │   └── _noise_more
│   ├── our_manifest
│   │   ├── fine_tune
│   │   ├── for_general
│   │   └── sub_manifest
│   ├── sample_16000
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
│   │       ├── Restaurant
│   │       ├── Station
│   │       ├── Street
│   │       └── Train
│   ├── speech_data
│   │   ├── backward
│   │   ├── bed
│   │   ├── bird
│   │   ├── cat
│   │   └── dog
│   └── wav
│       ├── 20240716
│       └── 20240717
└── src
    ├── create_data
    ├── data_preprocessing
    ├── fine_tune
    └── vad_algorithm
