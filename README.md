# annoy_ltm

This repository contains an extension for the oobabooga-text-generation-webui application, introducing long-term memory to chat bots using the Annoy (Approximate Nearest Neighbors Oh Yeah) nearest neighbor vector database.

## Features

The `annoy_ltm` extension provides chat bots with a form of long-term memory. It leverages the efficient search algorithm of Annoy to retrieve similar vector representations from the history, allowing the bot to reference past interactions.

## Installation

This extension can be installed like any other extension to the oobabooga-text-generation-webui, with an additional requirement for the Spacy language model. Follow the instructions below:

1. Download and install the Spacy en_core_web_sm model. You can do this by running the `cmd_windows.bat` and then executing the following commands in the resulting cmd shell:

Windows WSL:

```bash
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
Linux:
In the environment you are using for Oobabooga-text-generation-webui, run the folowing command:

```bash
python -m spacy download en_core_web_sm
```
2. Follow the regular installation process for extensions to the oobabooga-text-generation-webui application.

3. Navigate to the annoy_ltm extension folder and run the following command to install the dependencies:
```bash
pip install -r requirements.txt
```


## Usage

Once the extension is enabled, it works automatically with no additional steps needed. You can configure its behavior by modifying the following parameters in the `settings.json` of the webui:

| Parameter                   | Description     | Default Value |
| --------------------------- | --------------- | ------------- |
| `annoy_output_dir`          | Directory where outputs are stored. | `"extensions/annoy_ltm/outputs/"` |
| `logger_level`              | Logging level, higher number results in more verbose logging. Maximum reasonable value for normal debugging is 3. | `1` |
| `memory_retention_threshold`| Retention threshold for memories. Lower values cause memories to retain longer, potentially at the cost of stack overflow and irrelevant memory retention. Ranges from 0-1. | `0.68` |
| `full_memory_additional_weight`| Additional weight for the full memory. Smaller values result in higher weight. Ranges from 0-1. | `0.5` |
| `num_memories_to_retrieve`  | Number of related memories to retrieve for the full message and every keyword group generated from the message. Higher values can cause significant slowdowns. | `5` |
| `keyword_grouping`          | Number to group keywords into. Higher values make it harder to find an exact match, potentially improving context relevance at the cost of memory retrieval. | `4` |
| `maximum_memory_stack_size` | Maximum size for the memory stack, preventing overflow. | `50` |
| `prompt_memory_ratio`       | The ratio of the prompt after character context is applied that will be dedicated for memories. | `0.4` |
| `vector_dim_override` | Override value for the hidden layer dimension of your loaded model, Use if you encounter issues with the generated embeddings not matching the dimensionality of the annoy index. `-1` is disabled. | `-1` |

These parameters allow you to tune the operation of `annoy_ltm` to best suit your specific use-case.

## Support

For any issues or queries, please use the Issues tab on this repository.

## Docker
Hey you! Yeah you about to install some random project extension code into your non-dockerized oobabooga instance! Don't you know that's dangerous?  I highly recommend you check out the docker setup for oobabooga-text-generation-webui before randomly installing anything and do your due dilligance by reading through the extension code! You got that kind of time.
https://github.com/oobabooga/text-generation-webui#alternative-docker
