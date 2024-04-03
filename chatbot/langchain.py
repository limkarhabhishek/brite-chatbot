import logging
import os

from django.conf import settings
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rest_framework.exceptions import APIException

logger = logging.getLogger("chatbot")


class LangChain:
    """
    The LangChain class is responsible for loading the relevant blog,
    processing it, and then generating answers.
    """

    def __init__(self) -> None:
        """
        Initializes the LangChain class.
        """

        self.prompt = hub.pull("rlm/rag-prompt")

        self.template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
        self.custom_rag_prompt = PromptTemplate.from_template(self.template)

    def get_files(self):
        """
        Fetches files from the transcripts directory and stores them in the 'file_paths' attribute.
        """
        transcripts_directory = (
            "/home/linux/Chatbot/Django/brite_chatbot/chatbot/transcripts"
        )

        # Get the list of .txt files in the directory
        txt_files = [
            os.path.join(transcripts_directory, file)
            for file in os.listdir(transcripts_directory)
            if file.endswith(".txt")
        ]

        self.file_paths = txt_files

    def load_file_data(self):
        """
        Loads the contents of the text files and stores them in the 'docs' attribute.
        """
        self.docs = []
        for file in self.file_paths:
            loader = TextLoader(file)
            doc = loader.load()[0]
            self.docs.append(doc)

    def split_texts(self):
        """
        Splits the texts into smaller chunks of text and stores them in the 'splits' attribute.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.splits = text_splitter.split_documents(self.docs)

    def create_vector_store(self):
        """
        Creates a vector store from the texts and stores it in the 'vector_store' attribute.
        """
        self.vector_store = Chroma.from_documents(
            documents=self.splits,
            embedding=OpenAIEmbeddings(openai_api_key=self.openai_api_key),
        )

    @staticmethod
    def format_docs(docs):
        """
        Formats a list of documents as a single string.

        Args:
            docs (List[Document]): A list of Document objects.

        Returns:
            str: The formatted string.
        """
        return "\n\n".join(doc.page_content for doc in docs)


class OpenAIChatbot(LangChain):
    """Class for working with OpenAI's GPT-3 API to generate answers"""

    def __init__(self):
        """Initialize the class"""
        super().__init__()
        self.openai_api_key = settings.OPENAI_API_KEY

    def create_retriever(self):
        """Create a retriever from the vector store"""
        self.retriever = self.vector_store.as_retriever()

    def create_openai_llm(self):
        """Create an OpenAI LLM using the API key"""
        self.openai_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125", openai_api_key=self.openai_api_key
        )

    def create_open_ai_rag_chain(self):
        """Create a RAG chain for OpenAI generation of answers"""
        self.open_ai_rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.openai_llm
            | StrOutputParser()
        )

    def create_open_ai_rag_chain_with_sources(self):
        """Create a RAG chain for OpenAI generation of answers, along with source information"""

        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self.format_docs(x["context"]))
            )
            | self.prompt
            | self.openai_llm
            | StrOutputParser()
        )

        self.open_ai_rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    def invoke_open_ai_chat_with_source(self, question):
        """Invoke the OpenAI RAG chain with sources, returning an answer and a list of sources"""
        try:
            output = self.open_ai_rag_chain_with_source.invoke(question)
        except Exception as e:
            raise APIException({"detail": f"Something went wrong! {e}"})
        answer = output.get("answer")
        source_list = []
        for source in output["context"]:
            if source.metadata["source"] not in source_list:
                source_list.append(source.metadata["source"])

        return {"answer": answer, "source_list": source_list, "status": 200}

    def initialize_prerequisites(self):
        """Initialize all prerequisites for using the OpenAI Chatbot"""
        logger.info("Initializing prerequisites")

        self.get_files()
        logger.info(f"Got {len(self.file_paths)} files")
        self.load_file_data()
        logger.info(f"Loaded {len(self.file_paths)} file pieces")
        self.split_texts()
        logger.info(f"Split {len(self.splits)} texts")
        self.create_vector_store()
        logger.info("Created vector store")
        self.create_retriever()
        logger.info("Created retriever")
        self.create_openai_llm()
        logger.info("Created OpenAI LLM")
        # self.create_open_ai_rag_chain()
        # logger.info("Created OpenAI RAG chain")
        self.create_open_ai_rag_chain_with_sources()
        logger.info("Created OpenAI RAG chain with sources")
        logger.info("Initialization complete")
