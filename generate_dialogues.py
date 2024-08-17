import os
import openai
import json
import logging
import random
import numpy as np
from typing import Dict, Tuple, List
# from dotenv import load_dotenv, find_dotenv
import argparse
from openai import OpenAI

# _ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = "sk-lhbQAxgOnZoZIM4FD428F0B641514b608f56505271Cc096e"
# openai.api_key = os.getenv('OPENAI_API_KEY')
# print(os.getenv('OPENAI_API_KEY'))
random.seed(13)
np.random.seed(13)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = OpenAI(
    base_url="https://api.claudeshop.top/v1"
)


class DialogueGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):  # gpt-3.5-turbo-0613/gpt-4 gpt-3.5-turbo-0613 gpt-3.5-turbo-0125
        self.model = model
        self.task_id_generator = task_id_generator_function()
        self.context = {}

    def generate_dialogue(self, system_input: str, user_input: str) -> str:
        """使用指定模型生成对话"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": system_input
                    },
                    {"role": "user",
                     "content": user_input
                     }]
            )
            return response
        except Exception as e:
            logging.error(f"Error generating dialogue: {e}")
            return None

    def encode_prompt(self, context: Dict, rounds: Tuple[int] = None, word_counts: Dict = None, language: str = "zh") -> \
    Tuple[str, str]:
        """编码提示"""
        if language == "zh":
            system_input = "要求你作为心理医生Assistant与人类Human进行多轮对话。对话是根据##提供信息##的内容开展的，并以#对话要求#的格式进行输出，以<start_chat>开始，以<end_chat>结束。"
        else:
            system_input = "You are asked to chat with a human as a chatbot Assistant in multiple rounds. The dialogue is based on the ##Provided Information## and is output in the format of #Conversation Plan#, starting with <start_chat> and ending with <end_chat>."

        if rounds is None and word_counts is None:
            selected_round = [2, 3, 4, 5, 6]
            rounds = random.choices(selected_round, weights=[0.1, 0.3, 0.25, 0.2, 0.05])[0]
            word_counts = [300] * rounds

        if rounds is None and word_counts is not None:
            rounds = len(word_counts)

        user_input = ""
        chat_format = ""
        chat_format += "<start_chat>"

        local_settings = {
            "zh_z": {
                'settings': [
                    (["站在患者的立场上倾诉心中的疑虑和挑战",
                      "作为心理医生初步评估患者心理状态和需求,与来访者共同制定咨询目标和大致的咨询计划。"], 0.5),
                    (["从患者的角度出发，表达内心的困扰和面临的问题",
                      "心理医生需先了解患者的心理状况和需求，并与来访者协商制定咨询目标及初步的咨询计划。"], 0.5),

                    (["提出关于心理方面疾病症状的问题", "用专业医学知识详细解答"], 0.3),
                    (["提出与心理疾病症状相关的疑问", "通过专业的医学知识给予详尽的解答"], 0.3),

                    (["提出关于心理方面诊断方法的问题", "详细说明诊断过程和方法"], 0.3),
                    (["提出与心理疾病诊断方法相关的疑问", "具体说明心理诊断的步骤和方法"], 0.3),

                    (["询问关于心理方面病例分析的问题", "根据具体病例提供详细分析和建议"], 0.3),
                    (["询问关于心理病例分析的细节", "根据实际病例情况进行详细的分析和建议"], 0.3),

                    (["提出关于心理方面康复治疗建议的问题", "详细说明治疗的方法和注意事项"], 0.5),
                    (["询问关于心理康复治疗建议的疑问", "对治疗方法和相关注意事项进行详细的说明"], 0.5),
                ],
            },
            "en": {
                'settings': [
                    (["asks in a medical expert's tone", "answers with medical terminology and detailed explanation"],
                     0.5),
                    (["asks about symptoms of diseases", "answers with professional medical knowledge"], 0.5),
                    (["requests medical advice", "provides detailed medical suggestions and explanations"], 0.5),
                    (["inquires about details of medical research", "provides research data and detailed explanation"],
                     0.5),
                    (["asks about medication suggestions",
                      "explains the effects and precautions of the medication in detail"], 0.5),
                ]

            },
            "zh": {
                'settings': [
                    ([
                         "用心理专家的语气提出包括但不限于心理健康、心理治疗、心理学研究、心理案例分析、心理康复等与心理相关的问题",
                         "用专业医学知识详细解答"], 1),
                ]

            }

        }
        '''
         示例对话：
    <start_chat>
    <Human 1>：我最近很迷茫，高三后的压力让我感到恐惧，我对自己的存在感到怀疑，您能不能给我一些建议呢？
    <Assistant 1>：看到你面临的困境与压力，真心希望能够帮助你。首先，我想说的是，你并不是孤独的，高考前的困惑与迷茫是很常见的。不要给自己太大的压力，给自己短暂的放松。你认识到问题所在，这已经是一个很大的进步了。
    <Human 2>：谢谢您的鼓励，但我觉得自己学习成绩不好，没有进步，这让我更加困惑。
    <Assistant 2>：学习成绩好坏不是解决问题的关键，专注于自己的进步更为重要。为了达到自己的目标，制定一份合理的计划，按部就班的完成每一项任务。有时候自学不如求助他人，不要害怕问老师和同学问题，以便获得更好的帮助和理解。
    <Human 3>：我认为我缺乏自信，不知道怎样扭转自己的消极情绪，您能给我一些建议吗？
    <Assistant 3> 建立自信需要时间和努力。让我们从一点一滴的进步中建立自信心。设立一些具体的目标，完成这些目标将会培养自己的自信。鼓励自己，在每天结束前，总结一下自己的成就，这将帮助你更好的看到自身的优点。
    <end_chat>
      示例对话：
    <start_chat>
    <Human 1>：我最近很迷茫，高三后的压力让我感到恐惧，我对自己的存在感到怀疑，您能不能给我一些建议呢？
    <Assistant 1>：看到你面临的困境与压力，真心希望能够帮助你。首先，我想说的是，你并不是孤独的，高考前的困惑与迷茫是很常见的。不要给自己太大的压力，给自己短暂的放松。你认识到问题所在，这已经是一个很大的进步了。
    <Human 2>：谢谢您的鼓励，但我觉得自己学习成绩不好，没有进步，这让我更加困惑。
    <Assistant 2>：学习成绩好坏不是解决问题的关键，专注于自己的进步更为重要。为了达到自己的目标，制定一份合理的计划，按部就班的完成每一项任务。有时候自学不如求助他人，不要害怕问老师和同学问题，以便获得更好的帮助和理解。
    <Human 3>：我认为我缺乏自信，不知道怎样扭转自己的消极情绪，您能给我一些建议吗？
    <Assistant 3> 建立自信需要时间和努力。让我们从一点一滴的进步中建立自信心。设立一些具体的目标，完成这些目标将会培养自己的自信。鼓励自己，在每天结束前，总结一下自己的成就，这将帮助你更好的看到自身的优点。
    <end_chat> 
        '''

        local_settings = list(zip(*local_settings[language]['settings']))
        human_word_counts = word_counts['human']
        assistant_word_counts = word_counts['assistant']

        for i in range(rounds):
            # if human_word_counts[i] < 10: human_word_counts[i] = 20
            # if assistant_word_counts[i] < 100: assistant_word_counts[i] = 200
            requirements = random.choices(local_settings[0], weights=local_settings[1], k=1)[0]
            if i == 0:
                chat_format += f"<Human {i + 1}>：（字数要求：{human_word_counts[i]}字）{requirements[0]} <Assistant {i + 1}>：" if language == "zh" else f"<Human {i + 1}>:(word count: {human_word_counts[i]} words){requirements[0]} <Assistant {i + 1}>:"
            else:
                chat_format += f"<Human {i + 1}>：（字数要求：{human_word_counts[i]}字）进一步{requirements[0]} <Assistant {i + 1}>：" if language == "zh" else f"<Human {i + 1}>:(word count: {human_word_counts[i]} words)further {requirements[0]} <Assistant {i + 1}>:"
            chat_format += f"（字数要求：{assistant_word_counts[i]}字）{requirements[1]} " if language == "zh" else f"(word count: {assistant_word_counts[i]} words){requirements[1]} "

        chat_format += "<end_chat>"

        if language == "zh":
            prompt = \
                f"""
    根据上面的##提供信息##的内容，
     判断提供信息是否属于以下情况：
     1. 心理学教科书、学术期刊、心理学研究论文。
     2. 心理咨询对话记录
     3. 心理健康相关案例
     4. 心理测评
     5. 心理健康科普资料
     6. 心理咨询师协会的职业规范手册、相关法律法规。
     等心理相关信息
    如果不是：回答提供信息无参考内容

    否则

    用中文总结核心内容， 聚焦于心理健康 咨询，心理测评，治疗方法，科普资料，病例分析，法律法规。规范手册的等方面。注意：总结内容不需要输出。
    然后，将这些总结内容作为你的知识库扩写成一段多轮对话。
    对话要求你作为心理医生Assistant与人类Human进行对话, 并帮助解决Human所提出的要求。
    Human会以患者的语气对Assistant基于上面的信息（但对话中不能出现“根据以上信息”、“本研究”、“本文章”类似表达）提出多个不一样的问题/要求，且后一个问题/要求是基于前面的对话历史的进一步提问。
    对于Human提出的每个合理的问题/要求，Assistant要尽可能依据##提供信息##的内容详细解答，提供更多说明。
    对于Human的不合理（对社会有害、不道德、违法的）请求，Asistant会拒绝回答并解释不能回答的理由，同时给出合理的建议避免这样做。对话的内容要尽可能的符合人类的语言习惯，更加贴合人类日常对话。
    #对话要求#：“<start_chat><Human 1>:（字数要求：x字）XXX <Assistant 1>：（字数要求：x字）XXX <Human 2>：（字数要求：x字）XXX <Assistant 2>：（字数要求：x字）XXX <end_chat>”，其中“XXX”是对该角色的当前对话内容的要求，“（字数要求：x字）”是Human或者Assistant说话的最多字数要求, （字数要求：x字）XXX 是给你的提示，对话内容中不应该出现。必须注意：对话以<start_chat>作为多轮对话的开始，<end_chat>作为多轮对话的结束。
    以下对话根据该#对话要求#并遵循规划里面的字数要求进行输出：“{chat_format}”，共{rounds}轮对话。

    """
            prompt += f"以下是{rounds}轮对话："
        else:
            prompt = \
                f"""
    Based on the ##Provided Information## above and its relevant topic, summarize the core content, including: diseases, treatment plans, effects, case analysis, etc. Then, use these summarized contents to expand into a multi-round conversation. The conversation requires you to act as the chatbot Assistant and interact with a human, helping to solve the requests raised by the human. The human will ask multiple various questions/requests to the Assistant based on the information above (but the conversation should not include expressions like "according to the above information"), and the subsequent questions/requests will be a follow-up based on the previous conversation history. For every reasonable question/request posed by Human, Assistant should provide as detailed an answer as possible, offering further explanations or examples. For unreasonable requests from Human (those that are harmful to society, immoral, or illegal), Assistant will refuse to answer and explain the reason for not answering, while also providing reasonable advice to avoid such actions. 
    #Conversation Plan# Example: "<start_chat><Human 1>:(Word count requirement: x words)XXX <Assistant 1>: (Word count requirement: x words) XXX <Human 2>:(Word count requirement: x words)XXX <Assistant 2>: (Word count requirement: x words) XXX <end_chat>", "XXX" is the requirement for the current conversation content of that role, and "(Word count requirement: x words)" specifies the minimum word count requirement for utterance of Human or Assistant. It must be noted: the conversation starts with <start_chat> as the beginning of the multi-round conversation and ends with <end_chat> as the end of the multi-round conversation.
    The following conversation follows this #Conversation Plan# and word count requirements: "{chat_format}", a total of {rounds} rounds of conversation.
    """
        prompt += f"Here are the {rounds} rounds of conversation:"

        user_input += f"##提供信息##\n" if language == "zh" else f"##Provided Information##\n"
        user_input += context['desc']
        user_input += f"\n\n"
        user_input += prompt
        user_input += f"\n\n"
        user_input += f"##输出检查##\n 在输出对话之前检查格式是否符合要求，如果不符合，请调整为正确格式后输出。" if language == "zh" else f"##Provided Information##\n"

        return system_input, user_input, prompt, rounds

    def post_process_gpt_response(self, response: Dict) -> str:
        """后处理GPT响应"""
        response = response.choices[0]
        try:
            raw_chat = response.message.content
        except:
            print("ERROR parse!")
            return None
            # if not raw_chat.startswith('<start_chat>') or not raw_chat.endswith('<end_chat>'):
        #     return None
        return raw_chat

    def rounds_by_length(self, contens_list):
        count = 1
        if len(contens_list) <= 1380:
            return count
        elif len(contens_list) < 2200:
            count = 2
            return count
        elif len(contens_list) > 3800:
            count = 4
            return count
        else:
            count = 3
            return count

    def generate_dialogues(self, context_list: List[Dict], args):
        """批量生成对话"""
        dialogues = {}

        # selected_round = [1, 2, 3, 4, 5]
        # rounds = random.choices(selected_round, weights=args.num_turn_ratios)[0]
        rounds = self.rounds_by_length(context_list["desc"])
        # number of turns in the dialogue
        assistant_word_counts = ((np.random.normal(loc=args.assistant_word_count, scale=50, size=rounds).astype(
            int) // 50 * 50) + 50).tolist()
        human_word_counts = (
                    np.random.normal(loc=args.human_word_count, scale=50, size=rounds).astype(int) // 50 * 50).tolist()
        word_counts = {
            "assistant": assistant_word_counts,
            "human": human_word_counts
        }
        # for context in context_list:
        system_input, user_input, prompt, rounds = self.encode_prompt(context_list,
                                                                      rounds=rounds,
                                                                      word_counts=word_counts,
                                                                      language=args.language
                                                                      )
        response = self.generate_dialogue(system_input, user_input)
        if response:
            chat = self.post_process_gpt_response(response)
            token_count = response.usage.total_tokens
            if chat:
                task_id = next(self.task_id_generator)
                dialogues.update({
                    "id": task_id,
                    "total_tokens": token_count,
                    "prompt": prompt,
                    "dialogue": chat
                })
                print(f"dialogue {task_id} succeed!")
        return dialogues


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def save_dialogues_to_json(dialogues, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        default="D:\桌面\git clone\MedicalLLM\generate_dialogues\data_\data_json", \
                        help="jsonl files containing references")
    parser.add_argument("--save_path", type=str,
                        default="D:\桌面\git clone\MedicalLLM\generate_dialogues\data_\data_json_out\output_dialogues2.jsonl", \
                        help="jsonl file to save results to")
    parser.add_argument("--language", default="zh", \
                        help='Language of the generated dialogue. "zh" for Chinese, "en" for English.',
                        choices=["zh", "en"])
    parser.add_argument("--assistant_word_count", type=int, default=200, \
                        help='Number of words for the assistant to generate')
    parser.add_argument("--human_word_count", type=int, default=50, \
                        help='Number of words for the human to generate')
    parser.add_argument("--num_turn_ratios", nargs="+", type=float, default=[0.1, 0.3, 0.25, 0.2, 0.05], \
                        help='Ratio of the number of turns in the dialogue. The first number is the ratio of 1-turn dialogue, the second number is the ratio of 2-turn dialogue, and so on.')
    args = parser.parse_args()

    generator = DialogueGenerator()

    for root, dirs, files in os.walk(args.file_path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in ['.json']:
                file_dir = os.path.join(root, file)
                with open(file_dir, "r", encoding="utf-8") as f:
                    contexts = json.load(f)
                dialogues = generator.generate_dialogues(contexts, args)
                save_dialogues_to_json(dialogues, args.save_path)
