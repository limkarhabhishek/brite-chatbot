from chatbot.models import OpenAIFeedback
from django.contrib import admin


# Register your models here.
class CustomOpenAIFeedback(admin.ModelAdmin):
    """Customized admin panel"""

    list_display = ("id", "question", "generated_answer", "dislike")
    list_filter = ("id", "question", "dislike")


admin.site.register(OpenAIFeedback, CustomOpenAIFeedback)
 