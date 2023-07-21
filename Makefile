install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

offline:
	python3 examen_indexar.py archive temp

