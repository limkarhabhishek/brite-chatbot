from chatbot.models import OpenAIFeedback
from rest_framework.serializers import ModelSerializer


class OpenAIFeedbackRequestSerializer(ModelSerializer):
    class Meta:
        model = OpenAIFeedback
        fields = '__all__'