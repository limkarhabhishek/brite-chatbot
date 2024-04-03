from django.urls import path

from chatbot.views import OpenAIView, OpenAIFeedbackView


urlpatterns = [
    path("open-ai", OpenAIView.as_view(), name="open-ai"),
    path('openai-feedback', OpenAIFeedbackView.as_view(), name='openai-feedback-list'),
]
