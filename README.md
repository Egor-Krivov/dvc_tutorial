# Tutorial: Predict Mobile Device change 
Прогноз смены мобильного девайса в течение месяца 

# Описание задачи

Задача:
- бинарная классификация
- шаг - 1 месяц
- целевые метрики - Lift, Precision@K, Recall@K

Таблицы:

**target.feather**

    — user_id - ID пользователя
    — month - округление date до последнего дня месяца
    — target - 1/0

**user_features.feather**

    — user_id - ID пользователя
    — month - месяц, за который собраны фичи
    — feature_N - конкретная фича (всего 50 фичей)

Валидация:

 - 4 фолда (по месяцам), агрегации метрик по фолдам - mean, std, min, max


Доп. информация и комментарии:

- Для target = 0 date = None! Таргет - это событие смены девайса, у этого события есть дата. Если событие не произошло, то и даты нет. Нам надо будет эти таргеты смотреть в течение месяца 
- Сколько раз юзер может выполнить целевое действие?  Скорее всего, это будет 1 раз в какой-то месяц за весь период


# How to Run 

## 1. Fork / Clone repo
- fork to your personal repo 
- clone to you local machine


## 2. Use a virtual environment

Сreate and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies
```bash
pip install -r venv_requirements.txt
```

Add virtual environment to Jupyter Notebook
```bash
python -m ipykernel install --user --name=venv
``` 

Run Jupyter Notebook 
```bash
jupyter notebook
```

To deactivate virtual environment: 
```bash
deactivate 
```



## 3. Or run with Docker 

Create config/.env
```bash
GIT_CONFIG_USER_NAME=<git user>
GIT_CONFIG_EMAIL=<git email>
```
example:

```.env
GIT_CONFIG_USER_NAME=mnrozhkov
GIT_CONFIG_EMAIL=mnrozhkov@gmail.com
```

Build
```bash
ln -sf config/.env 
docker-compose build
```

Run 
```bash
docker-compose up
```

Open tutorial notebook:
- open http://localhost:8888 in browser;


## 4. Get dataset
- Download data from [storage](https://yadi.sk/d/HibDNFMf3oTEoA)
- Put data to `data/raw`


## 5. Create link to data/ folder (optional)
```bash
cd noteboks/ 
ln -sf ../data
```
