# Instructing and Prompting Large Language Models for Explainable Cross-domain Recommendations

## Structure:
This repository is organized into three main folders:

- `GPT-CDR`: This folder contains code for using the GPT API to retrieve recommendations.
- `open-models-train`: This folder houses the code for training open-source large language models (LLMs) like LLaMA or Mistral, as well as code to set up the training environment.
- `open-models-inference`: This folder contains code for performing inference on the fine-tuned LLM you trained in the previous step.

## Using GPT as CDR
### Prerequisites
1. **Create a virtual environment**: To isolate dependencies, it's recommended to create a virtual environment using `venv`.
2. **Install requirements**: Activate the virtual environment and run `pip install -r GPT-CDR/requirements.txt` to install the necessary libraries.
3. **Obtain OpenAI API Key**: You'll need an OpenAI API key to access and use the GPT model. Refer to the OpenAI documentation for obtaining a key.

### Inference
**Insert OpenAI API Key**: Insert your OpenAI API key in the `src/model/GPT_RecSys.py` file.
**Run GPT_RecSys.py**: Execute the script using the following command:
```
python GPT_RecSys.py --base-domain "books" --target-domain "movies" --test-case books530_cds1030 --batch-size 1
```
- Replace `"books"` and `"movies"` with the actual base and target domains for your recommendation task
- Choose a valid `--test-case` from the options listed in `conf.py`. 
- Adjust `--batch-size` if needed, keeping in mind request per time constraints of open AI plans.

## Train an open model
### Prerequisites
1. **Create a virtual environment**: To isolate dependencies, it's recommended to create a virtual environment using `venv`.
2. **Install requirements**: Activate the virtual environment and run `pip install -r open-models-train/requirements.txt` to install the necessary libraries.

### Model training
Before training the open model, you'll need to configure the hyperparameters that control the training process. These hyperparameters are defined in the `open-models-train/training/train_conf.yaml` file. The parameters listed in this file correspond to those used in the research paper associated with the model. Finally, the script of training can be run `python open-models-train/training/main_train.py`


## Test a fine-tuned model
### Prerequisites
1. **Use the train virtual environment**: Make sure you are using the same virtual environment you created for training the model. This ensures you have the necessary libraries and dependencies installed to load the trained model.
2. **Use the model trained in the training step**: Specify the path to the trained model weights file you obtained from the training process. This file typically contains the learned parameters of the model.

### Model inference
Similar to model training, inference relies on the `config.yaml` file to define various settings like the data scenario and the model itself.  To run inference, execute the following command `python open-models-inference/main.py`
