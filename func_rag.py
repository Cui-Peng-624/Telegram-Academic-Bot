import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

from openai import OpenAI

# 官方API
# client = OpenAI()

# 第三方低价API
api_key = ""
api_base = ""
client = OpenAI(api_key=api_key, base_url=api_base)

import telebot

BOT_TOKEN = ""
bot = telebot.TeleBot(BOT_TOKEN)

from pinecone import Pinecone, ServerlessSpec 

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "telebot"  # change if desired
index = pc.Index(index_name)

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key, base_url=api_base) # 使用第三方低价API

# 用户输入命令后选择文档类别
@bot.message_handler(commands=['rag'])
def ask_field(message):
    """
    让用户选择领域并提问。
    """
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True) # 卧槽,牛逼啊,变成选择题了!
    fields = ['Education', 'Computer_Science', 'Medicine', 'Literature', 'Other']
    for field in fields:
        markup.add(field)
    bot.send_message(message.chat.id, "请选择一个领域:", reply_markup=markup)
    bot.register_next_step_handler(message, get_user_question)

def get_user_question(message):
    """
    获取用户的问题并开始处理查询。
    """
    selected_field = message.text
    bot.send_message(message.chat.id, "请输入您的问题:")
    bot.register_next_step_handler(message, lambda m: process_query(m, selected_field))

def process_query(message, field):
    """
    处理用户查询：在Pinecone中搜索相似内容并生成答案。
    """
    user_question = message.text
    namespace = field
    results = search_similar_documents(user_question, namespace)
    top_k_results = [res.page_content for res, _ in results]
    
    # 将用户问题与检索结果一同传给OpenAI生成最终答案
    final_answer = generate_answer(user_question, top_k_results)
    
    # 返回结果给用户
    bot.send_message(message.chat.id, final_answer, parse_mode="Markdown")

from langchain_pinecone import PineconeVectorStore

def search_similar_documents(query, namespace, top_k=5):
    """
    在Pinecone的特定namespace中查找与query最相似的文档内容。
    """
    vector_store = PineconeVectorStore(index=index, namespace=namespace, embedding=embeddings)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results

def generate_answer(user_question, documents):
    """
    将用户问题和检索到的文档内容传入OpenAI模型生成答案。
    """
    context = "\n".join(documents)
    user_prompt = f"用户问题: {user_question}\n相关文档:\n{context}\n请根据这些内容回答问题："

    # system prompt 用于指导模型回答
    system_prompt = (
        "你是一个帮助用户回答基于文档内容的问题的助手。"
        "当用户提出问题时，你应该根据提供的相关文档中的信息进行回答。"
        "如果文档中包含答案，尽可能引用文档内容并确保准确性。"
        "如果文档中没有明确的答案，请基于你对文档的理解提供推断，但避免过度猜测。"
    )
    
    messages_to_model = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 使用OpenAI生成最终答案
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages_to_model)
    # print(user_prompt)
    return completion.choices[0].message.content

# 启动Bot
bot.polling()

