from django.urls import path

from chatbot.views import OpenAIView


urlpatterns = [
    path("open-ai", OpenAIView.as_view(), name="open-ai"),
]
