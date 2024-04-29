import pandas as pd
import sys
sys.path.append("src/models")
import utilities as ut


SEPARATOR = "###################"
MISTRAL_ANSWERS = []

# -------------------
# Collect demo data
# -------------------
with open("src\\demo\\demo_mistral_answers.txt", "r") as file:
    mistral_file_content = file.read()
    mistral_original_answers = str(mistral_file_content).split(SEPARATOR)

# GPT-3.5 turbo
with open("src\\demo\\demo_gpt_answers.txt", "r") as gpt_file:
    gpt_file_content = gpt_file.read()
    GPT_ANSWERS = str(gpt_file_content).split(SEPARATOR)


# -------------------
# Extract recommendations
# -------------------
for mistral_answer in mistral_original_answers:
    MISTRAL_ANSWERS.append(ut.filter_llm_output(mistral_answer, model="mistral"))

print("===============")
print("MISTRAL ANSWERS")
print("===============")
for answer in MISTRAL_ANSWERS:
    print(answer)
    print("----------")
    
print("===========")
print("GPT ANSWERS")
print("===========")
for answer in GPT_ANSWERS:
    print(answer)
    print("----------")