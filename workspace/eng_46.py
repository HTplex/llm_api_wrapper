

import json
json_path = "/Users/htplex/Developer/ht/llm_api_wrapper/tmp/rag_data.json"
with open(json_path, 'r') as fp:
    data = json.load(fp)
print(data[0])
answers = [d["Content"] for d in data]
machine_scores = [d["machine_score"] for d in data]
print(machine_scores[:10])
print(answers[0])
print(len(answers))
print(machine_scores[0])
print(len(machine_scores))


prompt = "你是一个专业的四六级英语评分专家，请根据以下内容，给出评分，评分标准：\n"
standard = """
    翻译评分标准
    - 14分段(13-15分)：译文准确表达了原文的意思，用词贴切，行文流畅，基本无语言错误，仅个别错误。（1-4个单词层面小错误。
    In recent years, the northeast region of China has been greatly developing its ice and snow resources. For example, Harbin has utilized its abundant ice and snow resources to build an “ice and snow world” with its regional   characteristics, allowing tourists to enjoy the beauty of ice and snow while experiencing its unique folk culture. Nowadays, what used to be a fearful icey and snowy land is attracting tourists from all over the world, becoming a popular tourist destination. The ice and snow tourism is making greater and greater contribution to the development of the local economy.
    - 11分段(10-12分)：译文基本上表达了原文的意思，文字较连贯，但有少量语言错误。（ 5 ~ 8 个小错误，甚至包括语言的错误，包括信息传达不准确的这些错误。出现四个错误，但是错误有一点重也属于11分段）
    In recent years, China's northeast area is developing ice-and-snow resources rapidly For example, Harbin has created “the world of ice and snow" with special local features by making full use of its abundant ice-and-snow resources, which lets visitors enjoy the beauty of ice and snow while they were engagingy  in the unique local traditional folk culture. Nowadays, the frozen area with ice and snow, which used to make people frightened, areattracting visitors from everywhere, becoming an especially popular resort. The ice-and-snow traveling industry is making an increasing number of contributions to the local economic development.
    - 8分段(7-9分):译文勉强表达了一小部分原文的意思，连贯性差，有相当多的严重语言错误。（正确率大约一半。
    每一个句子磕磕绊绊的，甚至是结构上的很多的这么大的错，但是他传的有些地方你能读到他一些简单需要什么的，这大概反映出大概一半的意思，有句子架构。每一个句子都有错，有些句子甚至破了，但是只能表达出一半的意思。就是比较典型的 8 分
    - 5分段(4-6分):译文仅表达了一小部分原文的意思，连贯性差，有相当多的严重语言错误。
    改卷是大量的这种卷子，四级改卷翻译，基本上平均分可能还很难上 5 分，大概就是 5 分， 5.1 分。基本的词汇都拼写错误，只能够说小部分地传达了原文的意思，中间偶尔能读到两三个简单句子，表达了小部分的意思，这是非常典型的5分段。
    - 2分段:(1-3分):除个别词语或句子，绝大部分文字没有表达原文意思。
    一般的情况下 2 分段就基本上读不到句子，支离破碎的你能够读到一些完整的词组，那么两个、三个、四个都可以，只能够读到一点词组方面的信息。
    沾边给分，与原文相关的关键词构成词组1分 
    - 0分:未作答，或只有几个孤立的词，或译文与原文毫不相干。
    慎给0分！跨套卷:0分，答题区有翻译又有作文：只管翻译；无翻译只有作文，0分
"""
gt = "敦煌莫高窟数字展示中心于2014年开放，是莫高窟保护利用工程的重要组成部分。展示中心采用数字技术和多媒体展示方式，让游客在进入洞窟参观前了解莫高窟文化，欣赏其艺术经典，从而限制洞窟的开放数量，缩短游客在洞窟内的停留时间，减少对莫高窟的影响，确保这一世界文化遗产得到妥善保护和长期利用。"
golden = "The Digital Exhibition Center of Dunhuang Mogao Grottos,which opened in 2014,plays a vitat rolein the conservationand utilization project of the Mogao Grottos.Utilizing digi-tal technology and multimedia exhibition methods,thecenter enables visitorsto gain an understanding of the cul-tural significance of the Mogao Grottos and appreciate itsclassic arts before they enter the actual caves.This ap-proach not only limits the number of caves open to thepublic but also reduces the duration of visitors'stay inside,thereby minimizing the impact on the Mogao Grottos.This ensures the proper preservation and long-term utilizationof this world cultural heritagesite."
format = "不要输出任何解释，直接输出整型分数，分数范围是0-15分"

prompts = [f"{prompt}\n{standard}\n 翻译题目：{gt}\n 参考答案：{golden}\n 学生答案: {x} \n {format}" for x in answers]
print(prompts[0])



from os.path import expanduser
import json
import sys
import os
sys.path.append(os.path.abspath('..'))
from llmw.wrapper_v2 import LLMW
from pprint import pprint
    
if __name__ == "__main__":
    api_keys_path = expanduser("~/.ssh/api_keys.json")
    api_key= json.load(open(api_keys_path))["OPENROUTER-API-KEY"]["api_key"]
    llmw = LLMW(api_key, model="deepseek-r1", max_concurrency=1000)
    prompts = prompts[:]
    answers = llmw.batch_call(prompts)
    with open("eng_46_answers.json", 'w') as fp:
        json.dump(answers, fp, sort_keys=True, indent=4, ensure_ascii=False)

    print(len(answers))
    
    pprint(answers)


