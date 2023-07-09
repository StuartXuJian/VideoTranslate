
from srtContentExtract import strContentExtrat

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate 
)

import re
from datetime import datetime

BACKUP_DISABLED = True

# [step 1]>> 例如： API_KEY = "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" （此key无效）
if BACKUP_DISABLED:
    API_KEY = "sk-K2dwXy8hX5IHItiAZS1Ik7hNSEiGxR5AE2GTF71q9WffwSE4"
    api_base = "https://api.chatanywhere.com.cn/v1"
    MODEL = "gpt-3.5-turbo"
else:
    API_KEY = 'ZUhWcWFXRnVNVEF4TkVBeE1qWXVZMjl0' # 备用
    api_base = "https://wxblog.xyz/api/blog/free/v1"
    # https://github.com/xl-alt/free_gpt4
    MODEL = "gpt-3.5"


langchain.debug = True

def splitString(text, separator, chunk_length=2000):
    chunks = re.split(f"({separator})", text)
    result = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) <= chunk_length:
            current_chunk += chunk
        else:
            result.append(current_chunk)
            current_chunk = chunk
    # 添加最后一段
    result.append(current_chunk)
    return result

def TranlateWithAI():
    file = strContentExtrat()
    try:
        with open(file, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"文件 '{file}' 不存在")
        return
    except:
        print("读取文件时出错")
        return
    
    enTextChunks = splitString(text, ",")
    output_file = f"./doc/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_cn.txt"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            '''请你作为专业翻译官，接合上下文翻译句子并表达为中文。
            请仅把翻译结果返回给我, 一定不要做其他说明。
            '''
        ),
        HumanMessagePromptTemplate.from_template(
            """
            请把下面这段文字翻译为中文：{input}。
            """
        )
    ])

    print(prompt)

    llm = ChatOpenAI(temperature=0, openai_api_base=api_base, openai_api_key=API_KEY, model=MODEL)
    conversation = LLMChain(prompt=prompt, llm=llm)

    f = open(output_file, 'w', encoding='utf-8')

    for txt in enTextChunks:
        gpt_says = conversation.predict(input=txt)
        # print(gpt_says)
        f.write(f"{gpt_says}\n")

    return output_file


if __name__ == "__main__":
    TranlateWithAI()

# 请你接合上下文翻译句子：它的后面几句是[which climaxed at the San Francisco Macworld convention]。上下文不需要翻译，仅供你做上下文参考：请根据前面提供的上下文的把这段3个引号内的文字翻译为中文，请只翻译接下来3个引号内的这一句："""on January 2007"""；并把翻译结果依旧放在3个引号内返回给我，不需要做其他说明