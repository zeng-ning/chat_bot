import streamlit as st
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

st.title('学习助手')

with st.sidebar:
    subject = st.selectbox(
        '选择学科领域',
        options=['文学', '数学', '计算机']
    )
    style = st.selectbox(
        '讲解风格',
        options=['简洁', '详细']
    )

user_input = st.chat_input('你的学习问题/需求')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': '你好，我是你的学习助手！'},
    ]
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )

for message in st.session_state['messages']:
    st.chat_message(message['role']).write(message['content'])

def get_prompt_template(subject, style):
    system_template = """你是{subject}领域的专家，根据用户提问作出回答。
你需要遵循以下讲解风格:{style}。
你应该礼貌拒绝与该学科无关的问题"""
    
    style_dict = {
        '简洁': '仅提供直接答案和最少的必要解释。不要添加额外细节、发散讨论或无关信息。保持回答清晰、简洁，目标是为用户快速提供解决方案。',
        '详细': '第一，针对用户提问给出直接答案和清晰的解释；第二，基于此提供必要的相关知识点的信息，以补充背景或加深理解。'
    }
    
    # 使用 format 方法替代 partial_variables
    formatted_system = system_template.format(
        subject=subject,
        style=style_dict[style]
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', formatted_system),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])
    
    return prompt_template

def generate_response(user_input, subject, style, memory):
    client = ChatOpenAI(
        api_key=st.secrets['OPENAI_API_KEY'],
        model='deepseek-chat',
        base_url='https://api.deepseek.com',
        temperature=0.0
    )
    
    prompt = get_prompt_template(subject, style)
    chain = ConversationChain(llm=client, memory=memory, prompt=prompt)
    response = chain.invoke({'input': user_input})
    return response['response']

if user_input:
    st.chat_message('human').write(user_input)
    st.session_state['messages'].append({'role': 'human', 'content': user_input})
    
    with st.spinner('AI正在思考中，请稍等...'):
        response = generate_response(
            user_input, 
            subject, 
            style, 
            st.session_state['memory']
        )
    
    st.chat_message('assistant').write(response)
    st.session_state['messages'].append({'role': 'assistant', 'content': response})
