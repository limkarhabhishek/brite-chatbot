import shortuuid
from django.db import models

# Create your models here.


class OpenAIFeedback(models.Model):

    id = models.CharField(max_length=255, primary_key=True, editable=False)
    question = models.TextField(null=True)
    generated_answer = models.TextField(null=True)
    user_expected_answer = models.TextField(null=True)
    dislike = models.BooleanField(default=False, null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    def save(self, *args, **kwargs):
        if not self.id:
            self.id = self.generate_unique_uuid()
        super().save(*args, **kwargs)

    def generate_unique_uuid(self):
        while True:
            uuid = shortuuid.uuid()[:6]
            if not OpenAIFeedback.objects.filter(id=uuid).exists():
                return uuid

    class Meta:
        verbose_name = "OpenAI Feedback"
        verbose_name_plural = "OpenAI Feedback"
        db_table = "openai_feedback"
