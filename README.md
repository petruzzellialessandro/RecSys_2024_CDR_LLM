# Instructing and Prompting Large Language Models for Explainable Cross-domain Recommendations

## Structure:
This repository is organized into three main folders:



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