install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

indexar:
	rm -rf temp &&\
		python3 examen_indexar.py archive temp

buscar:
	python examen_buscar.py archive q temp similar con-input
