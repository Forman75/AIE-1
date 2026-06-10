# Удобные команды проекта.
# Использование: make <цель>

.PHONY: help install train serve test docker-build docker-run clean

help:            ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:         ## Установить зависимости
	pip install --upgrade pip
	pip install -r requirements.txt

train:           ## Обучить модели и сохранить артефакты
	python -m src.train

serve:           ## Запустить REST API сервис локально
	python -m src.service

test:            ## Запустить тесты
	pytest tests

docker-build:    ## Собрать Docker-образ
	docker build -t churn-prediction .

docker-run:      ## Запустить сервис в Docker
	docker run -p 8000:8000 churn-prediction

clean:           ## Удалить кэши и артефакты обучения
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
	rm -f artifacts/model.joblib artifacts/metrics.json artifacts/model_comparison.csv
	rm -rf artifacts/plots artifacts/experiments
