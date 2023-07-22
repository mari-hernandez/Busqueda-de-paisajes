install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

indexar:
	rm -rf temp &&\
		python3 examen_indexar.py archive temp

buscar:
	python examen_buscar.py archive new_metadata.json q temp similar con-input

buscar-toda-carpeta:
	python examen_buscar.py archive new_metadata.json q temp similar
