from typing import Any, Dict, List
import os
import time
import openai
import logging
import random

import requests
from transformers import GPT2TokenizerFast,AutoTokenizer, AutoModelForCausalLM
import http.client
import json

avoid_keywords = ["one", "two", "three", "1", "2", "3", "a", "he", "she", "i", "we", "you", "it", "this", 
        "that", "the", "those", "these", "they", "me", "them", "what", "him", "her", "my", "which", "who", "why", 
        "your", "my", "his", "her", "ours", "our", "could", "with", "whom", "whose"]
os.environ["http_proxy"] = 'http://172.31.226.133:7890'
os.environ["https_proxy"] = 'http://172.31.226.133:7890'

class GPT3():
    def __init__(self, model="gpt-3.5-turbo", interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4, max_prompt_length=4096):
        self.model = model
        self.interval = interval
        self.timeout = timeout
        self.base_timeout = timeout
        self.patience = patience
        self.exp = exp
        self.max_prompt_length = max_prompt_length
        self.max_interval = max_interval
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.AUTHORIZATION = 'Bearer sk-8DHk7uanpqrJwEXfOdXIN3wuvUShznTBEBkyHZJQ6tcNT4uy'

    def call(
        self, prompt, temperature=1.0, top_p=1.0, max_tokens=64, n=1, 
        frequency_penalty=0, presence_penalty=0, stop=["Q:"], rstrip=False,
        **kwargs):

        #openai.api_key = os.environ.get('OPENAI_API_KEY', None)

        # check if exceeding len limit
        input_len = len(self.tokenizer(prompt).input_ids)
        if input_len + max_tokens >= self.max_prompt_length:
            logging.warning("OpenAI length limit error.")
            return [""] * n

        # stop words
        if isinstance(stop, List):
            pass
        elif isinstance(stop, str):
            stop = [stop]

        if rstrip:
            prompt = prompt.rstrip()

        retry_interval_exp = 1 
        t1 = time.time()

        while True and retry_interval_exp <= self.patience:
            try:
                # conn = http.client.HTTPSConnection("api.chatanywhere.com.cn")

                if self.model == "gpt-3.5-turbo": # chat completion
                    payload = json.dumps({
                        "model" : self.model,
                        "messages" : [
                            {
                                "role" : "user",
                                "content" : prompt
                            }
                        ],
                        "temperature" : temperature,
                        "max_tokens" : max_tokens,
                        "n" : n,
                        "top_p" : top_p,
                        "frequency_penalty" :frequency_penalty,
                        "presence_penalty" : presence_penalty,
                        "stop" : stop
                    })
                    headers = {
                        'Authorization' : self.AUTHORIZATION,
                        'User-Agent' : 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type' : 'application/json'
                    }
                    # response = openai.ChatCompletion.create(model=self.model,
                    #                                     messages=messages,
                    #                                     temperature=temperature,
                    #                                     max_tokens=max_tokens,
                    #                                     n=n,
                    #                                     top_p=top_p,
                    #                                     frequency_penalty=frequency_penalty,
                    #                                     presence_penalty=presence_penalty,
                    #                                     stop=stop,
                    #                                     request_timeout=self.timeout # timeout!
                    #                                     )
                    response = requests.post(url='https://api.chatanywhere.com.cn/v1/chat/completions', headers=headers, data=payload, proxies={'https': os.environ["http_proxy"], 'http': os.environ["http_proxy"]})
                    # conn.request("POST", "/v1/chat/completions", payload, headers)
                    # res = conn.getresponse()
                    data = json.loads(response.text)
                    # print(data.decode("utf-8"))
                    candidates = data["choices"]
                    candidates = [candidate["message"]["content"] for candidate in candidates]

                else: # text completion
                    headers = {
                        'Authorization': self.AUTHORIZATION,
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    payload = json.dumps({
                        "model": self.model,
                        "prompt" : prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "n": n,
                        "top_p": top_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty,
                        "stop": stop
                    })
                    # response = openai.Completion.create(model=self.model,
                    #                                     prompt=prompt,
                    #                                     temperature=temperature,
                    #                                     max_tokens=max_tokens,
                    #                                     n=n,
                    #                                     top_p=top_p,
                    #                                     frequency_penalty=frequency_penalty,
                    #                                     presence_penalty=presence_penalty,
                    #                                     stop=stop,
                    #                                     request_timeout=self.timeout # timeout!
                    #                                     )
                    response = requests.post(url='https://api.chatanywhere.com.cn/v1/completions', headers=headers, data=payload, proxies={'https': os.environ["http_proxy"], 'http': os.environ["http_proxy"]})
                    # conn.request("POST", "/v1/completions", payload, headers)
                    # res = conn.getresponse()
                    data = json.loads(response.text)
                    # print(data.decode("utf-8"))
                    candidates = data["choices"]
                    candidates = [candidate["message"]["content"] for candidate in candidates]

                t2 = time.time()
                logging.info(f"{input_len} tokens, {t2-t1} secs")

                return candidates

            # except openai.error.RateLimitError as e:
            except Exception as e:
                # logging.warning("OpenAI rate limit error. Retry")
                logging.warning(e)
                # Exponential backoff
                time.sleep(max(self.max_interval, self.interval * (self.exp ** retry_interval_exp)))
                retry_interval_exp += 1
        
        return None
    

if __name__ == "__main__":
    gpt3 = GPT3()
    
    # messages = []
    # for i in range(100):
    #     messages.append(f"what is the sum of {random.randint(1000, 10000)} and {random.randint(1000, 10000)}?")
    # predictions = gpt3.async_call(prompt=messages)

    for i in range(100):
        message = f"what is the sum of {random.randint(1000, 10000)} and {random.randint(1000, 10000)}?"
        predictions = gpt3.call(prompt=message)
        print(message, predictions)