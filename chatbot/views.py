import logging

from chatbot.langchain import OpenAIChatbot
from chatbot.models import OpenAIFeedback
from chatbot.serializers import OpenAIFeedbackRequestSerializer
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
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
    pagination_class = PageNumberPagination  # Set the pagination class

    def get(self, request, format=None):
        """
        GET request that retrieves paginated OpenAI feedback requests.

        Returns:
            Paginated list of OpenAI feedback requests in JSON format.
        """
        # Order the queryset with descending order based on the id field (UUID)
        feedbacks = OpenAIFeedback.objects.order_by("-created_at")

        # Get paginated queryset using pagination class
        paginator = self.pagination_class()
        paginated_queryset = paginator.paginate_queryset(feedbacks, request)

        serializer = OpenAIFeedbackRequestSerializer(paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)

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

    def delete(self, request):
        """
        Delete all records in the OpenAIFeedback model and return a response.

        :param request: The request object passed to the function.
        :return: A Response object with the detail message indicating that all records have been deleted.
        """
        OpenAIFeedback.objects.all().delete()
        return Response({"detail": "All the records have been deleted"})
