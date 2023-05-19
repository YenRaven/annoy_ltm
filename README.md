# oobabooga-webui annoy_ltm
annoy long term memory experiment for oobabooga/text-generation-webui

## Requirements
You will need to download the spacy `en_core_web_sm` model to use this extenstion
run cmd_windows.bat and run the following in the resulting cmd shell

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```

## Docker
Hey you! Yeah you about to install some random project extension code into your non-dockerized oobabooga instance! Don't you know that's dangerous?  I highly recommend you check out the docker setup for oobabooga-text-generation-webui before randomly installing anything and do your due dilligance by reading through the extension code! It's one file, you got that kind of time.
https://github.com/oobabooga/text-generation-webui#alternative-docker
