import json
import time
from typing import Optional, Dict, Any, List, Tuple
import os

from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from huggingface_hub import login

# Log into Hugging Face (needed if you're using a gated model)
login(token="token")

class RAGService:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        token: str = "token",
        db_path: str = "chroma_db"
    ):
        # Load the LLM pipeline
        self.model_name = model_name
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            token=token
        )

        # Set up vector store location
        self.db_path = db_path
        self._init_vector_db()

    def _init_vector_db(self):
        # Try to load Chroma from disk; if that fails, create a new one
        try:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=FastEmbedEmbeddings()
            )
        except Exception:
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=FastEmbedEmbeddings()
            )

    def ingest_message(
        self,
        conversation_id: str,
        text: str,
        role: str = "user"
    ):
        # Add a message (either from user or assistant) to the vector DB
        doc = Document(
            page_content=text,
            metadata={
                "conversation_id": conversation_id,
                "doc_type": "message",
                "role": role,
                "timestamp": time.time()
            }
        )
        doc = filter_complex_metadata([doc])
        self.vector_store.add_documents(doc)
        self.vector_store.persist()

    def retrieve_conversation(self, conversation_id: str) -> List[Tuple[str, str]]:
        # Pull out all messages in a conversation, ordered by time
        col = self.vector_store._collection
        filter_ = {
            "$and": [
                {"conversation_id": conversation_id},
                {"doc_type": "message"}
            ]
        }
        result = col.get(where=filter_, include=["documents", "metadatas"])

        docs = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            role = meta["role"]
            ts = meta["timestamp"]
            docs.append((role, text, ts))

        docs.sort(key=lambda x: x[2])  # Sort by timestamp
        return [(role, text) for (role, text, _) in docs]

    def build_prompt(self, conversation_data: List[Tuple[str, str]], new_event: str) -> str:
        # Construct the full prompt the model will see — includes context and the new user message
        system_text = (
            "You are a role play agent. You will receive an event and must act like it's "
            "happening to you. You respond as the character, describing your next move. "
            "Respond in the format: The agent does <action> etc.\n"
        )

        lines = [system_text]
        for role, text in conversation_data:
            if role == "assistant":
                lines.append(f"Assistant: {text}")
            else:
                lines.append(f"User: {text}")

        # Add the new event at the end
        lines.append(f"User: {new_event}\nAssistant:")
        return "\n".join(lines)

    def generate_response(
        self, 
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = 128
    ) -> str:
        # Use the pipeline to generate text based on the full prompt
        outputs = self._pipeline(
            prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens
        )
        return outputs[0]["generated_text"]

    def handle_event(
        self,
        conversation_id: str,
        event_details: str,
        llm_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Step 1: Save the new user message
        self.ingest_message(conversation_id, event_details, role="user")

        # Step 2: Pull conversation history
        convo_data = self.retrieve_conversation(conversation_id)

        # Step 3: Create a prompt using past messages + this new one
        prompt = self.build_prompt(convo_data, event_details)

        # Step 4: Get model parameters from input
        temperature = float(llm_params.get("temperature", 0.7))
        top_p = float(llm_params.get("top_p", 0.9))
        top_k = int(llm_params.get("top_k", 40))

        # Step 5: Generate raw model output
        raw_gen = self.generate_response(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        # Step 6: Clean up the output — remove prompt and redundant lines
        answer_only = raw_gen[len(prompt):].strip()

        if answer_only.startswith("Assistant:"):
            answer_only = answer_only[len("Assistant:"):].strip()

        filtered_lines = []
        for line in answer_only.split("\n"):
            if line.strip().startswith("User:"):
                continue
            if line.strip() == event_details.strip():
                continue
            filtered_lines.append(line)

        llm_response = "\n".join(filtered_lines).strip()

        # Step 7: Save the assistant’s response
        self.ingest_message(conversation_id, llm_response, role="assistant")

        # Step 8: Return final output
        return {
            "conversation_id": conversation_id,
            "llm_response": llm_response
        }

if __name__ == "__main__":
    # Load test input from file
    input_filename = "test_event.json"
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pull out conversation ID, event, and any model params
    conversation_id = data["conversation_id"]
    event_details = data["event_details"]
    llm_params_list = data.get("llm_parameters", [0.7, 0.9, 40])

    llm_params = {
        "temperature": llm_params_list[0],
        "top_p": llm_params_list[1],
        "top_k": llm_params_list[2]
    }

    # Set everything up and run the event through the pipeline
    rag_service = RAGService(
        model_name="meta-llama/Llama-3.2-1B",
        db_path="chroma_db"
    )

    result = rag_service.handle_event(conversation_id, event_details, llm_params)

    # Save the final result to an output file
    output_filename = f"{conversation_id}_output.json"
    with open(output_filename, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2, ensure_ascii=False)
