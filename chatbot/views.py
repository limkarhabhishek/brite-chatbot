import logging

from chatbot.langchain import OpenAIChatbot
from rest_framework.response import Response
from rest_framework.views import APIView


openai = OpenAIChatbot()

logger = logging.getLogger(__name__)


class OpenAIView(APIView):
    initialized = False

    def get(self, request):

        if not OpenAIView.initialized:
            logger.info("Initializing OpenAI RAG chain")
            openai.initialize_prerequisites()
            OpenAIView.initialized = True

        # Get question from query parameters
        question = request.query_params.get("question", "")

        # Invoke question with source
        response = openai.invoke_open_ai_chat_with_source(question)

        # Return response
        return Response(response)
