import logging

from chatbot.langchain import OpenAIChatbot
from chatbot.models import OpenAIFeedback
from chatbot.serializers import OpenAIFeedbackRequestSerializer
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

openai = OpenAIChatbot()

logger = logging.getLogger(__name__)


class OpenAIView(APIView):
    initialized = False

    def get(self, request):
        # Get question from query parameters
        question = request.query_params.get("question", "")
        if not question:
            return Response(
                {"detail": "No question provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        if not OpenAIView.initialized:
            logger.info("Initializing OpenAI RAG chain")
            openai.initialize_prerequisites()
            OpenAIView.initialized = True

        # Invoke question with source
        response = openai.invoke_open_ai_chat_with_source(question)

        # Return response
        return Response(response)


class OpenAIFeedbackView(APIView):
    """
    View that handles GET and POST requests for OpenAI feedback requests.
    """

    serializer_class = OpenAIFeedbackRequestSerializer

    def get(self, request, format=None):
        """
        GET request that retrieves all OpenAI feedback requests.

        Returns:
            List of OpenAI feedback requests in JSON format.
        """
        feedbacks = OpenAIFeedback.objects.all()
        serializer = OpenAIFeedbackRequestSerializer(feedbacks, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        """
        POST request that creates a new OpenAI feedback request.

        Arguments:
            request: Django REST framework request object.

        Returns:
            Created OpenAI feedback request in JSON format.

        Raises:
            400 Bad Request: If the input JSON is invalid.
            201 Created: If the OpenAI feedback request is created successfully.
        """
        serializer = OpenAIFeedbackRequestSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
