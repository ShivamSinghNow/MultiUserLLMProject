# LocalLLMChat.py

import time
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class MultiUserChatMemory:
    """
    Stores chat messages as Documents in Chroma.
    """

    def __init__(self, model_name="qwen2.5", db_path="chroma_db"):
        self.llm = ChatOllama(model=model_name)
        self.prompt = ChatPromptTemplate([
            ("system", "You are a helpful assistant that remembers the user's chat history."),
            ("human",  "Conversation so far:\n{conversation}\n\nUser's new message:\n{query}"),
        ])
        self.db_path = db_path
        self._init_vector_db()

    def _init_vector_db(self):
        try:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=FastEmbedEmbeddings()
            )
        except:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=FastEmbedEmbeddings()
            )

    def store_message(self, user_id: str, role: str, text: str):
        from langchain.docstore.document import Document
        doc = Document(
            page_content=text,
            metadata={
                "user_id": user_id,
                "role": role,    # 'user' or 'assistant'
                "timestamp": time.time(),
                "doc_type": "message"
            },
        )
        self.vector_store.add_documents([doc])
        self.vector_store.persist()

    def get_user_conversation(self, user_id: str):
        """Load messages for this user from Chroma, sorted by timestamp."""
        collection = self.vector_store._collection

        # Use $and syntax if your Chroma supports it:
        filter_ = {
            "$and": [
                {"user_id": user_id},
                {"doc_type": "message"}
            ]
        }
        result = collection.get(
            where=filter_,
            include=["metadatas", "documents"]
        )

        docs = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            role = meta["role"]
            ts = meta["timestamp"]
            docs.append((role, text, ts))

        docs.sort(key=lambda x: x[2])  # sort by timestamp
        return [(role, text) for (role, text, _) in docs]

    def build_conversation_text(self, conversation):
        lines = []
        for role, text in conversation:
            if role == "user":
                lines.append(f"User: {text}")
            else:
                lines.append(f"Assistant: {text}")
        return "\n".join(lines)

    def ask(self, user_id: str, user_text: str) -> str:
        """Add user's message, retrieve entire convo, run LLM, store assistant reply."""
        # 1) Save user message
        self.store_message(user_id, "user", user_text)

        # 2) Build convo for LLM
        convo = self.get_user_conversation(user_id)
        convo_str = self.build_conversation_text(convo)

        # 3) Build chain
        chain = (
            {
                "conversation": lambda _: convo_str,
                "query": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        assistant_reply = chain.invoke(user_text)

        # 4) Store assistant's reply
        self.store_message(user_id, "assistant", assistant_reply)

        return assistant_reply

    def load_user_conversation_as_tuples(self, user_id: str):
        """Return conversation as (text, is_user) for the UI."""
        conv = self.get_user_conversation(user_id)  # [(role, text)]
        result = []
        for (role, txt) in conv:
            is_user = (role == "user")
            result.append((txt, is_user))
        return result

    def clear_user_history(self, user_id: str):
        """Delete all messages for this user."""
        # Use the same $and approach:
        filter_ = {
            "$and": [
                {"user_id": user_id},
                {"doc_type": "message"}
            ]
        }
        self.vector_store.delete(where=filter_)
        self.vector_store.persist()
