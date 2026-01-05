# 2.7 使用 LangChain 的聊天机器人

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/2.7%20Chatbots%20with%20LangChain/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/2.7%20Chatbots%20with%20LangChain/)

本章介绍使用 LangChain 构建和优化对话聊天机器人——一个将语言模型连接到检索系统以进行动态问答的工具包。我们采取实用路线：设置环境并加载文档，构建向量存储，选择高级检索策略，并添加对话记忆，以便机器人保持上下文并自信地回答后续问题。对话机器人改变数据交互：它们跟踪并记住对话，而不是独立的回合，LangChain 的模块化架构允许你逐步插入加载器（80+格式）、分块、嵌入、语义搜索、自我查询和上下文压缩。一个重要的细节是早期环境和变量设置：可观察性和仔细的关键字处理可以加快调试和操作。然后我们组装核心——对话检索链——结合语言模型、检索器和记忆，并展示如何缓冲记忆保留消息序列并将其与新问题一起传递，以保持对话的自然和连贯。

首先初始化环境并 API 密钥，以安全地使用云 LLM 并准备接口：

```py
# Import environment and API helpers
import  os
from  dotenv  import load_dotenv, find_dotenv

# Ensure Panel is available for interactive apps
import  panel  as  pn
pn.extension()

# Load environment variables (including the OpenAI API key)
_ = load_dotenv(find_dotenv())

# OPENAI_API_KEY is read by integrations automatically; no direct assignment required 
```

选择一个语言模型版本并将其固定用于演示：

```py
# Choose a model version
import  datetime
current_date = datetime.datetime.now().date()
language_model_version = "gpt-3.5-turbo"
print(language_model_version) 
```

现在连接嵌入和向量存储以进行基线问答：加载/索引文档、检索相关片段并准备模型以生成答案。然后定义一个提示模板并组装一个将使用您的检索器并构建上下文答案的检索 QA 链：

```py
# Embeddings and vector store
from  langchain.vectorstores  import Chroma
from  langchain_openai  import OpenAIEmbeddings

# Replace 'your_directory_path' with the directory where you will persist embeddings
persist_directory = 'your_directory_path/'
embedding_function = OpenAIEmbeddings()
vector_database = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# Query the vector store
search_question = "What are the key subjects covered in this course?"
top_documents = vector_database.similarity_search(search_question, k=3)
print(f"Relevant documents found: {len(top_documents)}")

# Initialize a chat model and try a simple greeting
from  langchain_openai  import ChatOpenAI
language_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
greeting_response = language_model.invoke("Greetings, universe!")
print(greeting_response)

# Prompt for concise, helpful answers
from  langchain.prompts  import PromptTemplate
prompt_template = """
Use the following pieces of context to answer the question at the end. If you're unsure about the answer, indicate so rather than speculating. 
Try to keep your response within three sentences for clarity and conciseness. 
End your answer with "thanks for asking!" to maintain a polite tone.

Context: {context}
Question: {question}
Helpful Answer:
"""
qa_prompt_template = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# RetrievalQA chain
default_question = "Does this course require understanding of probability?"
from  langchain.chains  import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    language_model,
    retriever=vector_database.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt_template}
)
qa_result = qa_chain({"query": default_question})
print("Result:", qa_result["result"]) 
```

### 实现具有记忆的问答对话检索链

本节面向机器学习工程师、数据科学家和构建理解并保留对话上下文的问答系统的开发者。重点是整合来自 LangChain 的记忆组件到对话检索链中。

#### 配置对话历史记录的记忆

使用`ConversationBufferMemory`以便系统记住上下文。它存储消息历史并允许引用先前的回合以进行相关的后续操作。

```py
# Conversation memory
from  langchain.memory  import ConversationBufferMemory

conversation_history_memory = ConversationBufferMemory(
    memory_key="conversation_history",
    return_messages=True
) 
```

#### 组装对话检索链

将语言模型、文档检索器和对话记忆结合以在对话上下文中回答问题。

```py
from  langchain.chains  import ConversationalRetrievalChain

document_retriever = vector_database.as_retriever()

question_answering_chain = ConversationalRetrievalChain.from_llm(
    llm=language_model,
    retriever=document_retriever,
    memory=conversation_history_memory
) 
```

#### 处理问题和生成答案

设置完成后，链可以处理问题并使用保存的对话历史作为上下文来生成答案。

```py
initial_question = "Is probability a fundamental topic in this course?"
initial_result = question_answering_chain({"question": initial_question})
print("Answer:", initial_result['answer'])

follow_up_question = "Why are those topics considered prerequisites?"
follow_up_result = question_answering_chain({"question": follow_up_question})
print("Answer:", follow_up_result['answer']) 
```

### 构建基于文档的问答聊天机器人

本部分提供了一种端到端的指南，介绍了一个基于文档内容回答问题的聊天机器人。它涵盖了加载文档、分割文本、嵌入和组装对话检索链。

#### 初始设置和导入

导入 LangChain 组件以进行嵌入、文本分割、内存搜索、文档加载、对话链和聊天模型。

```py
from  langchain_openai  import OpenAIEmbeddings
from  langchain.text_splitter  import RecursiveCharacterTextSplitter
from  langchain.vectorstores  import DocArrayInMemorySearch
from  langchain.document_loaders  import TextLoader, PyPDFLoader
from  langchain.chains  import ConversationalRetrievalChain
from  langchain_openai  import ChatOpenAI 
```

#### 加载和处理文档

加载文档，将它们分成可管理的块，生成嵌入，并准备向量存储；然后返回一个可用于对话检索链的现成链。

```py
def  load_documents_and_prepare_database(file_path, chain_type, top_k_results):
  """
 Load documents from a file, split into manageable chunks, generate embeddings,
 and prepare a vector database for retrieval.

 Args:
 - file_path: Path to the document file (PDF, text, etc.).
 - chain_type: Conversational chain type to use.
 - top_k_results: Number of top results to retrieve.

 Returns:
 - A conversational retrieval chain ready to answer questions.
 """
    # Load documents using a loader appropriate for the file type
    document_loader = PyPDFLoader(file_path)
    documents = document_loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    document_chunks = text_splitter.split_documents(documents)

    # Embed chunks and build the vector store
    embeddings_generator = OpenAIEmbeddings()
    vector_database = DocArrayInMemorySearch.from_documents(document_chunks, embeddings_generator)

    # Build the retriever
    document_retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={"k": top_k_results})

    # Create the conversational retrieval chain
    chatbot_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model='gpt-4o-mini', temperature=0), 
        chain_type=chain_type, 
        retriever=document_retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )

    return chatbot_chain 
```

为了方便，添加一个由 UI 代码使用的薄包装器：

```py
def  load_db(document_path, retrieval_type, top_k_results):
    return load_documents_and_prepare_database(document_path, retrieval_type, top_k_results) 
```

继续创建一个聊天机器人和基于面板的用户界面：导入 Panel (`pn`) 和 Param (`param`)，然后定义一个类，该类封装文档加载、查询处理和历史记录。

```py
import  panel  as  pn
import  param 
```

定义一个聊天机器人类，该类存储历史记录，形成答案，并允许交换基本文档。

```py
class  DocumentBasedChatbot(param.Parameterized):
    conversation_history = param.List([])  # (question, answer) pairs
    current_answer = param.String("")      # Latest answer
    database_query = param.String("")      # Query sent to the document DB
    database_response = param.List([])     # Retrieved source documents

    def  __init__(self, **params):
        super(DocumentBasedChatbot, self).__init__(**params)
        self.interface_elements = []  # UI elements for the conversation
        self.loaded_document = "your_document.pdf"  # Replace with your PDF path
        self.chatbot_model = load_db(self.loaded_document, "retrieval_type", 4)  # Initialize the bot model 
```

添加文档加载：`load_document` 检查用户文件（或使用默认文档），当源文件更改时重新加载知识库并清除历史记录。

```py
 def  load_document(self, upload_count):
        if upload_count == 0 or not file_input.value:
            return pn.pane.Markdown(f"Loaded document: {self.loaded_document}")
        else:
            file_input.save("temp.pdf")
            self.loaded_document = file_input.filename
            self.chatbot_model = load_db("temp.pdf", "retrieval_type", 4)
            self.clear_conversation_history()
        return pn.pane.Markdown(f"Loaded document: {self.loaded_document}") 
```

处理用户轮次：`process_query` 将轮次发送到模型，更新历史和 UI，并显示源片段。

```py
 def  process_query(self, user_query):
        if not user_query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.chatbot_model({"question": user_query, "chat_history": self.conversation_history})
        self.conversation_history.extend([(user_query, result["answer"])])
        self.database_query = result["generated_question"]
        self.database_response = result["source_documents"]
        self.current_answer = result['answer']
        self.interface_elements.extend([
            pn.Row('User:', pn.pane.Markdown(user_query, width=600)),
            pn.Row('Assistant:', pn.pane.Markdown(self.current_answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        input_field.value = ''  # Clear input
        return pn.WidgetBox(*self.interface_elements, scroll=True) 
```

为了透明度，显示最后的数据库查询和检索到的源文档。

```py
 def  display_last_database_query(self):
        if not self.database_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown("Last database query:", style={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("No database queries yet"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown("Database query:", style={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.database_query)
        )

    def  display_database_responses(self):
        if not self.database_response:
            return
        response_list = [pn.Row(pn.pane.Markdown("Vector DB search result:", style={'background-color': '#F6F6F6'}))]
        for doc in self.database_response:
            response_list.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*response_list, width=600, scroll=True) 
```

可选地，显示当前的聊天历史以快速检查。

```py
 def  display_chat_history(self):
        if not self.conversation_history:
            return pn.WidgetBox(pn.Row('Chat:', pn.pane.Str('No messages yet.')), scroll=True)
        items = []
        for q, a in self.conversation_history:
            items.append(pn.Row('User:', pn.pane.Markdown(q, width=600)))
            items.append(pn.Row('Assistant:', pn.pane.Markdown(a, width=600, style={'background-color': "#FAFAFA"})))
        return pn.WidgetBox(*items, width=650, scroll=True) 
```

不要忘记重置：`clear_conversation_history` 清除当前的对话上下文。

```py
 def  clear_conversation_history(self, count=0):
        self.conversation_history = [] 
```

结果是一个连贯的方法：你设置环境并配置密钥，加载文档并组装向量存储，添加高级检索（自查询、压缩、语义搜索），集成对话内存，并构建一个模型、检索器和内存共同工作的对话检索链。示例和代码展示了如何将步骤形成一个可工作的机器人；多亏了 LangChain 的模块化，结果易于扩展和调试。

## 理论问题

1.  设置 LangChain 聊天机器人开发环境需要哪些组件？

1.  保持对话历史如何提高聊天机器人的功能？

1.  文档块是如何转换为嵌入的，为什么？

1.  为什么自查询、压缩和语义搜索是有用的？

1.  对话检索链如何结合模型、检索器和内存？

1.  对话缓冲内存如何帮助维护对话上下文？

1.  配置 LangChain 中用于语义搜索的向量存储的步骤是什么？

1.  为什么需要管理环境变量和 API 密钥？

1.  LangChain 的检索方法的模块化如何增加开发灵活性？

1.  为什么选择合适的 LLM 版本很重要？

## 实践任务

1.  从字符串列表创建并填充向量存储（使用存根嵌入函数`embed_document`）。

1.  实现语义搜索 (`perform_semantic_search`)：嵌入查询，找到最近的文档，返回其索引。

1.  将对话历史添加到`Chatbot`类和一个`respond_to_query`方法中（通过存根`generate_response`生成）。

1.  组装一个使用存根（`LanguageModel`/`DocumentRetriever`/`ConversationMemory`）的简化对话检索链。

1.  在`Chatbot`中添加方法以追加和重置历史记录；在生成时包含历史记录。

1.  文档问答：加载一个字符串，将其分割，创建嵌入，构建向量存储，运行语义搜索，并生成答案（允许使用存根）。

1.  将内存集成到检索链中（使用任务 5-6 的扩展）。

1.  构建一个用于与`Chatbot`聊天的简单 CLI：发送查询，打印答案，查看/重置历史记录。

### 切换面板 UI 变体和仪表板

以下是一个额外的聊天机器人类和一个可立即使用的面板仪表板，它反映了俄罗斯版本的用户界面部分。

```py
import  panel  as  pn
import  param

class  ChatWithYourDataBot(param.Parameterized):
    conversation_history = param.List([])
    latest_answer = param.String("")
    document_query = param.String("")
    document_response = param.List([])

    def  __init__(self, **params):
        super(ChatWithYourDataBot, self).__init__(**params)
        self.interface_elements = []
        self.default_document_path = "your_document.pdf"  # Replace with your PDF path
        self.chatbot_model = load_db(self.default_document_path, "retrieval_mode", 4) 
```

添加最小方法实现以支持绑定的 UI 操作和面板：

```py
 def  load_document(self, clicks):
        if not getattr(document_upload, 'value', None):
            return pn.pane.Markdown(f"Loaded document: {self.default_document_path}")
        document_upload.save("temp.pdf")
        self.default_document_path = document_upload.filename or self.default_document_path
        self.chatbot_model = load_db("temp.pdf", "retrieval_mode", 4)
        self.clear_history()
        return pn.pane.Markdown(f"Loaded document: {self.default_document_path}")

    def  process_query(self, user_query):
        if not user_query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.chatbot_model({"question": user_query, "chat_history": self.conversation_history})
        self.conversation_history.extend([(user_query, result.get("answer", ""))])
        self.document_query = result.get("generated_question", "")
        self.document_response = result.get("source_documents", [])
        self.latest_answer = result.get('answer', "")
        self.interface_elements.extend([
            pn.Row('User:', pn.pane.Markdown(user_query, width=600)),
            pn.Row('Assistant:', pn.pane.Markdown(self.latest_answer, width=600, style={'background-color': "#F6F6F6"}))
        ])
        user_query_input.value = ""
        return pn.WidgetBox(*self.interface_elements, scroll=True)

    def  display_last_database_query(self):
        if not self.document_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown("Last database query:", style={'background-color': "#F6F6F6"})),
                pn.Row(pn.pane.Str("No database queries yet"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown("Database query:", style={'background-color': "#F6F6F6"})),
            pn.pane.Str(self.document_query)
        )

    def  display_database_responses(self):
        if not self.document_response:
            return
        items = [pn.Row(pn.pane.Markdown("Vector DB search result:", style={'background-color': "#F6F6F6"}))]
        for doc in self.document_response:
            items.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*items, width=600, scroll=True)

    def  display_chat_history(self):
        if not self.conversation_history:
            return pn.WidgetBox(pn.Row('Chat:', pn.pane.Str('No messages yet.')), scroll=True)
        items = []
        for q, a in self.conversation_history:
            items.append(pn.Row('User:', pn.pane.Markdown(q, width=600)))
            items.append(pn.Row('Assistant:', pn.pane.Markdown(a, width=600, style={'background-color': "#FAFAFA"})))
        return pn.WidgetBox(*items, width=650, scroll=True)

    def  clear_history(self, *_):
        self.conversation_history = [] 
```

创建小部件并绑定操作：

```py
document_upload = pn.widgets.FileInput(accept='.pdf')
load_database_button = pn.widgets.Button(name="Load document", button_type='primary')
clear_history_button = pn.widgets.Button(name="Clear history", button_type='warning')
clear_history_button.on_click(ChatWithYourDataBot.clear_history)
user_query_input = pn.widgets.TextInput(placeholder='Type your question here…')

load_document_action = pn.bind(ChatWithYourDataBot.load_document, load_database_button.param.clicks)
process_query = pn.bind(ChatWithYourDataBot.process_query, user_query_input) 
```

组装标签和仪表板：

```py
# Optional: Add a conversation flow diagram image if available
# conversation_visual = pn.pane.Image('path/to/your/conversation_flow.jpg')

conversation_tab = pn.Column(
    pn.Row(user_query_input),
    pn.layout.Divider(),
    pn.panel(process_query, loading_indicator=True, height=300),
    pn.layout.Divider(),
)

database_query_tab = pn.Column(
    pn.panel(ChatWithYourDataBot.display_last_database_query),
    pn.layout.Divider(),
    pn.panel(ChatWithYourDataBot.display_database_responses),
)

chat_history_tab = pn.Column(
    pn.panel(ChatWithYourDataBot.display_chat_history),
    pn.layout.Divider(),
)

configuration_tab = pn.Column(
    pn.Row(document_upload, load_database_button, load_document_action),
    pn.Row(clear_history_button, pn.pane.Markdown("Clears the conversation for a new topic.")),
    pn.layout.Divider(),
    pn.Row(conversation_visual.clone(width=400)),
)

chatbot_dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourDat-Bot')),
    pn.Tabs(('Conversation', conversation_tab), ('DB queries', database_query_tab), ('Chat history', chat_history_tab), ('Setup', configuration_tab))
) 
```

这完成了与俄罗斯章节的 UI 一致性，同时保持了英文文本的清晰和地道。
