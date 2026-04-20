import openai
import time
import argparse
import random
from openai import OpenAI


def get_response_openai(question, instruction="You are a helpful assistant."):
    # import pdb; pdb.set_trace()
    model_name, key, url = ["gpt-4.1", "sk-v3vzSqMLo0TxJf77440c430e75B04a90A6D02fCb0506B0D4", "https://xiaoai.plus/v1"]
    # model_name, key, url = ["gpt-4o", "sk-b4VB6HsCGierg3d2623158C77bB84aA68087C3B920A46d4c", "https://member.xiaoai.one/v1"]
    client = OpenAI(api_key=key, base_url=url)

    message = [
        {"role": "system", "content": instruction}, 
        {"role": "user", "content": question},
    ]

    # import pdb; pdb.set_trace()
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=0.0,
                stream=False,
            )
            # import pdb; pdb.set_trace()
            response = completion.choices[0].message.content
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default="What is the capital of France?")
    args = parser.parse_args()

    response = get_response_openai(args.question)
    print(f"Q: {args.question}\nA: {response}")