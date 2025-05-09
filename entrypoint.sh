#!/bin/sh

# Exit on error
set -e

if [ "$ENV" = "development" ]; then
    exec flask --app app.py run --host=0.0.0.0
else
    exec gunicorn -w 4 -b 0.0.0.0:5000 app:create_app
fi