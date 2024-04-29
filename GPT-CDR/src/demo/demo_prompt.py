import pandas as pd
import sys
sys.path.append("src/models")
from ScenarioGenerator import ScenarioGenerator

CDS_PATH = "data\\processed\\Books2_5\\books_2_5__1.csv"
MOVIES_PATH = "data\\processed\\Books2_5\\movies_5_10.csv"

COLUMNS_TYPES =  {
        'user_id': 'string',
        'item_id': 'string',
        'rating': 'string',
        'timestamp': 'string',
        'title': 'string',
        'brand': 'string',
        'category': 'string'
    }


# -------------------
# Collect demo data
# -------------------
cds = pd.read_csv(CDS_PATH, dtype=COLUMNS_TYPES)
movies = pd.read_csv(MOVIES_PATH, dtype=COLUMNS_TYPES)


# -------------------
# Demo prompt
# -------------------
user_ids = cds["user_id"].unique().tolist()

print("----------------------------------")
print("Running experiments for scenarios")
print("----------------------------------")

scenario_generator = ScenarioGenerator(scenario="zero-shot", is_free=False, full_log=True)
output_file_path = "src\demo\demo_prompt_zero_shot.txt"
with open(output_file_path, "w") as output_file:
    for i in range(0, 10):
        system_prompt, sample_user_prompts, assistant_prompts, user_prompt, gt = scenario_generator.generate_scenario(output_file=output_file, start_index=i, user_ids=user_ids, has_explanation=False, 
                                                                                                base_domain="cds", base_domain_data=cds, target_domain="movies", target_domain_data=movies, 
                                                                                                max_suggestions=2)
        output_file.write(f"\nsystem_prompt: \n{system_prompt}")
        output_file.write(f"\nsample_user_prompts: \n{sample_user_prompts}")
        output_file.write(f"\nassistant_prompts: \n{assistant_prompts}")
        output_file.write(f"\nuser_prompt: \n{user_prompt}")
        output_file.write(f"\nGround truth: \n{gt}")
        output_file.write("\n-------------------")



scenario_generator = ScenarioGenerator(scenario="one-shot")
output_file_path = "src\demo\demo_prompt_one_shot.txt"
with open(output_file_path, "w") as output_file:
    for i in range(0, 6, 2):
        system_prompt, sample_user_prompts, assistant_prompts, user_prompt, gt = scenario_generator.generate_scenario(output_file=output_file, start_index=i, user_ids=user_ids, has_explanation=False,
                                                                                              base_domain="cds", base_domain_data=cds, target_domain="movies", target_domain_data=movies, 
                                                                                              max_suggestions=1)
        output_file.write(f"\nsystem_prompt: \n{system_prompt}")
        output_file.write(f"\nsample_user_prompts: \n{sample_user_prompts}")
        output_file.write(f"\nassistant_prompts: \n{assistant_prompts}")
        output_file.write(f"\nuser_prompt: \n{user_prompt}")
        output_file.write(f"\nGround truth: \n{gt}")
        output_file.write("\n-------------------")
    
scenario_generator = ScenarioGenerator(scenario="three-shot")
output_file_path = "src\demo\demo_prompt_three_shot.txt"
with open(output_file_path, "w") as output_file:
    for i in range(0, 8, 4):
        system_prompt, sample_user_prompts, assistant_prompts, user_prompt, gt = scenario_generator.generate_scenario(output_file=output_file, start_index=i, user_ids=user_ids, has_explanation=False,
                                                                                              base_domain="cds", base_domain_data=cds, target_domain="movies", target_domain_data=movies, 
                                                                                              max_suggestions=1)
        output_file.write(f"\nsystem_prompt: \n{system_prompt}")
        output_file.write(f"\nsample_user_prompts: \n{sample_user_prompts}")
        output_file.write(f"\nassistant_prompts: \n{assistant_prompts}")
        output_file.write(f"\nuser_prompt: \n{user_prompt}")
        output_file.write(f"\nGround truth: \n{gt}")
        output_file.write("\n-------------------")
