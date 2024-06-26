# Use an official Python runtime as a parent image
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the project code into the container
COPY . /app/

# Expose port 8000 to the outside world
EXPOSE 8000

# Run Django migrations and make migrations
RUN python manage.py makemigrations
RUN python manage.py migrate

# Check if superuser exists before creating
# RUN python -c "import os; \
#                from django.contrib.auth.models import User; \
#                User.objects.filter(username='admin').exists() or \
#                os.system('python manage.py createsuperuser --noinput --username admin --email admin@example.com')"

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
