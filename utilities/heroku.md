# Hosting Website on Heroku

> keep debug = True for testing while hosting

1. install heroku cli on your machine

2. authenticate your self
	```
	heroku login
	```

3. generate requirements.txt file

4. create a heroku app
	```
	heroku create app-name
	heroku open # open url
	``` 
5. push code to heroku
	```
	heroku config:set DISABLE_COLLECTSTATIC=1 # to avoid error collect all static files for you
	# in setting.py mention STATIC_ROOT variable where it will collect all static files
	git push heroku main
	``` 
6. Add Procfile
	```
	create a Procfile 
	# data inside the procfile
	web: gunicorn --pythonpath app app.wsgi
	```
7. Add Environment variables 
	```
	heroku config:set KEY=VALUE
	```

8. Add Databas using heroku addon
	```
	heroku addons
	heroku pg #get details about the plan
	heroku addons:create heroku-postgresql:hobby-dev
	sudo apt-get install libpq-dev python-dev
	pip install psycopg2
	pip install django-heroku
	```

9. Go To seting.py and add these 
	```
	#starts
	import django_heroku

	#end
	django_heroku.settings(locals())
	```

10. Migrate in heroku hosted app
	```
	heroku run python app/manage.py makemigrations
	heroku run python app/manage.py migrate

	OR

	heroku run bash
	now you have ssh into heroku terminal 
	```
