Guide to install dependencies for *Review Analyzer project:

!!! These are all for WINDOWS OS !!!

-- Install Python
1.Install latest version of Python from given link: https://www.python.org/ftp/python/3.8.0/python-3.8.0.exe

-- Install Django
1. Open command-line(cmd) and run following commands sequentially:
2. curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
3. python get-pip.py 
4. python -m pip install Django

-- Install TextBlob
1. pip install -U textblob

-- Install BeautifulSoup
1. pip install beautifulsoup4
2. pip install requests

-- Install libraries
1. pip install django-crispy-forms
2. pip install image
3. pip install --upgrade django-cors-headers
4. pip install lxml
5. pip install --user -U nltk
6. python -m nltk.downloader stopwords
7. python -m nltk.downloader punkt
8. python manage.py makemigrations
9. python manage.py migrate

-- Run project
1. Open CMD
2. Go to the project directory
3. Type: python manage.py runserver
4. Take given IP adress(usually http://127.0.0.1:8000/)	and paste in browser
