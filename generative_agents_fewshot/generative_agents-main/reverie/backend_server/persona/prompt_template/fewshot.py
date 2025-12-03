import sys
import random
from fewshot_dataset import fewshot_dic
from gpt_structure import *

#run_gpt_prompt.py 내부로 이식된 코드입니다. 현재 사용하지 않습니다.



def get_fewshot_samples(key, n=1):

    '''
    get sample data for fewshot

    arg:
        key: MBTI type for sampling (ex. 'E', 'I', ...)
        n: number of sample data. it gets each n data for one mbti type, so total sample count will be 4n

    return:
        list of sample data
    '''

    subkeys = ['ENF', 'ENT', 'ESF', 'EST', 'INF', 'INT', 'ISF', 'IST']
    results = []

    for subkey in subkeys:
        if subkey in fewshot_dic[key] and fewshot_dic[key][subkey]:
            choices = random.sample(fewshot_dic[key][subkey], min(n, len(fewshot_dic[key][subkey])))
            results.extend(choices)

    return results

def run_gpt_few_shot(persona, original_text):
    def create_prompt(persona, original_text):
        # TODO: modify this part - add scratch.get_str_mbti() function to original persona class
        fewshot_samples = get_fewshot_samples(persona.scratch.get_str_mbti()) 

        prompt = 'Character Info\n'

        # part for real persona structure
        
        prompt += f'Name: {persona.scratch.get_str_name()}\n'
        prompt += f'Age: {persona.scratch.get_str_age()}\n'
        prompt += f'MBTI: {persona.scratch.get_str_mbti()}\n'   # TODO: mbti 관련 변수 및 함수 구현 필요
        prompt += f'Innate traits: {persona.scratch.get_str_innate()}\n'
        prompt += f'Learned traits: {persona.scratch.get_str_learned()}\n\n'
        prompt += f'Rephrase input sentence into a natural conversational style of {persona.scratch.get_str_firstname()}.\n'
        prompt += f'Use the speech patterns of other people whose MBTI character is similar to {persona.scratch.get_str_firstname()}.\n\n'

        for character_name, example_input, example_output in fewshot_samples:
            prompt += f'Input: {example_input}\n'
            prompt += f'{character_name}: {example_output}\n\n'

        prompt += 'CONSTRAINT: Output lengths must be similar to input length\n\n'
        prompt += f'Input: {original_text}\n'
        prompt += f'{persona.scratch.get_str_firstname()}: '    
        

        # code for test persona structure

        return prompt
    
    def __func_clean_up(gpt_response, prompt=""):
    
        """
        Cleans up the response from the GPT API for few-shot, persona-style conversational generation.

        - Strips leading/trailing whitespace and quotation marks
        - Removes unwanted newlines or excessive spaces
        - Ensures the response is a single line, matching the expected conversational style
        """

        # print for debug
        print("Few-shot GPT response (raw):")
        print(gpt_response)

        # Basic cleanup: strip, dedent, isolate first line if multi-line
        cr = gpt_response.strip()
        cr = cr.replace('\r', '').replace('\n', ' ').strip()
        if len(cr) == 0:
            return ""

        # Remove leading persona name
        persona_name = persona.get_str_firstname()  # Need to change: persona.scratch.get_str_firstname()
        if persona_name and cr.startswith(persona_name + ":"):
            cr = cr[len(persona_name)+1:].strip()

        # Remove quotation marks
        if cr and (cr[0] == '"' or cr[0] == "'"):
            cr = cr[1:]
        if cr and (cr[-1] == '"' or cr[-1] == "'"):
            cr = cr[:-1]

        # print for debug
        print("Few-shot GPT response (cleaned):")
        print(cr)

        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) == 0:
                return False
        except:
            return False
        return True

    def get_fail_safe(original_text):
        # print('Error: unsafe output') # for debugging
        return original_text
    
    # TODO: adjust parameter
    gpt_param = {"engine": "gpt-3.5-turbo", "max_tokens": 50, 
             "temperature": 0.8, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}

    prompt = create_prompt(persona, original_text)
    fail_safe = get_fail_safe(original_text)

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                   __func_validate, __func_clean_up)
    
    '''
    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                      prompt_input, prompt, output)
    '''

    return output



learned = 'Isabella Rodriguez is a cafe owner of Hobbs Cafe who loves to make people feel welcome. She is always looking for ways to make the cafe a place where people can come to relax and enjoy themselves.'
original_input = "I want to throw a Valentine's Day party today"
output = run_gpt_few_shot(persona, original_input)
print(output)
