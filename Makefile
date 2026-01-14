lint:
	pylint src
format:
	black src
check:
	pylint src
	black src --diff

N_TREES ?= 20

run:
	python src/main.py --n_trees $(N_TREES)