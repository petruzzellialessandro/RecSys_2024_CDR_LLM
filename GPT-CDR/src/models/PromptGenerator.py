import conf as conf
import utilities as ut

class PromptGenerator:
    def __init__(self, role):
        self.role = role
        self.prompt = ""


    def generate_prompt(self, has_explanation=False, has_suggest=True,
                        base_domain="", target_domain="", max_suggestions=5,
                        liked_items=[], candidate_items=[], suggested_items=[],
                        is_mistral_used=False):
        """
            Generate a prompt based on the specified role.

            Parameters:
            - has_explanation (bool): Whether the prompts should include explanations.
            - has_suggest (bool): Whether the user prompt should include a list of candidate items from which the LLM should recommend
            - base_domain (str): The name of the base domain.
            - target_domain (str): The name of the target domain.
            - max_suggestions (int): Maximum number of suggestions the LLM has to give.
            - liked_items (list): List of items liked by the user.
            - candidate_items (list): List of candidate items for recommendations.
            - suggested_items (list): List of suggested items by the assistant.
            - is_mistral_used (bool): Boolean value denoting if Mistral is currently used

            Returns:
            - str: The generated prompt.

            Roles:
            1. System role: Defines the task the language model has to carry out. If specified, it's possible to give it a hint.
            2. User role: Builds the user prompt containing liked items and candidate items for recommendations to get the recommendation.
            3. Assistant role: Used only for few-shot prompt scenarios, it formats the assistant's answer using a list of correctly recommended items.
        """

        if self.role == "system":
            self._generate_system_prompt(has_explanation, max_suggestions, target_domain)

        elif self.role == "user":
            self._generate_user_prompt(base_domain, liked_items, max_suggestions, target_domain, candidate_items,
                                       has_explanation, has_suggest, is_mistral_used)
        elif self.role == "assistant":
            self._generate_assistant_prompt(suggested_items, max_suggestions)

        elif self.role == "free_assistant":
            self._generate_free_assistant_prompt(target_domain, suggested_items, max_suggestions)

        return self.prompt


    # SYSTEM PROMPT ===========================================================================
    def _generate_system_prompt(self, has_explanation, max_suggestions, target_domain):
        self.prompt = conf.SYSTEM_PROMPT % (max_suggestions, target_domain)
        self.prompt += conf.SYSTEM_ANSWER_FORMAT % (max_suggestions, target_domain)

        if has_explanation:
            self.prompt += conf.SYSTEM_EXPLANATION_PROMPT


    # USER PROMPT ===========================================================================
    def _generate_user_prompt(self, base_domain, base_items_for_user, max_suggestions, 
                              target_domain, candidate_items,
                              has_explanation, has_suggestions, is_mistral=False):
        
        liked_items, disliked_items = ut.split_items_for_user(base_items_for_user)

        formatted_liked_items = ut.format_items_list(True, liked_items)
        formatted_disliked_items = ut.format_items_list(True, disliked_items)
        formatted_candidate_items = ut.format_items_list(False, candidate_items)

        # A list of candidate items is provided
        if has_suggestions:
            self.prompt = conf.USER_PROMPT % (base_domain, formatted_liked_items, 
                                              base_domain, formatted_disliked_items, 
                                              max_suggestions, target_domain,
                                              formatted_candidate_items)

        # No list of candidate items is provided
        else:
            self.prompt = conf.USER_PROMPT_NO_SUGGESTIONS % (base_domain, formatted_liked_items, 
                                                             base_domain, formatted_disliked_items, 
                                                             max_suggestions, target_domain)

        if is_mistral:
            self.prompt += conf.SYSTEM_ANSWER_FORMAT % (target_domain)

        # Ask the LLM for an explanation of the reasoning behind its suggestions
        if has_explanation:
                self.prompt += conf.SYSTEM_EXPLANATION_PROMPT

    # ASSISTANT PROMPT ===========================================================================
    def _generate_assistant_prompt(self, suggested_items, max_suggestions):
        ids = suggested_items["item_id"].head(max_suggestions).tolist()
        formatted_suggestions = ", ".join(str(item) for item in ids)
        self.prompt = conf.ASSISTANT_PROMPT % (formatted_suggestions)


    # ASSISTANT NO CANDIDATE ITEMS PROMPT ===========================================================================
    def _generate_free_assistant_prompt(self, target_domain, suggested_items, max_suggestions):
        formatted_suggestions = ut.format_items_list(False, suggested_items.head(max_suggestions), is_free_assistant=True)
        self.prompt = conf.ASSISTANT_PROMPT % (target_domain, formatted_suggestions)