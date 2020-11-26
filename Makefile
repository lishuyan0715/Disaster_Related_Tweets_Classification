api:
	python3 api.py

train:
	python3 bert/train.py

eda:
	python3 exps/eda.py

data_process:
	python3 exps/data_process.py

experiments:
	python3 exps/experiments.py

all: eda data_process experiments train

.PHONY: all