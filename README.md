# Классификация наличия сердечно-сосудистого заболевания

Лабораторная работа № 1: настройка окружения и разведочный анализ данных.

**Студент:** ФИО  
**Группа:** номер группы

## Описание проекта

Проект решает задачу бинарной классификации: по 13 клиническим признакам требуется определить наличие у пациента сердечно-сосудистого заболевания. Используется поднабор Cleveland из [Heart Disease Dataset (UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease), содержащий 303 наблюдения. Исходная целевая переменная `num` преобразуется в `target`: `0` — заболевание не выявлено, `1` — заболевание присутствует (`num > 0`).

Подробные результаты EDA будут добавлены после выполнения блокнота.

## Запуск

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd heart-disease-eda

python3 -m venv .venv_heart_disease
source .venv_heart_disease/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data" \
  -o data/heart_disease.csv

jupyter notebook eda/eda.ipynb
```

В Jupyter следует выполнить `Kernel -> Restart Kernel and Run All Cells`. Код блокнота корректно определяет корень проекта при запуске как из корня, так и из директории `eda`.

## Структура проекта

```text
heart-disease-eda/
├── data/                         # локальные данные (не входят в Git)
│   ├── heart_disease.csv         # исходный датасет
│   └── clean_dataset.pkl         # очищенный датасет
├── eda/
│   ├── eda.ipynb                 # воспроизводимый EDA
│   ├── *.png                     # статические графики
│   └── interactive_*.html        # интерактивный график
├── .gitignore
├── README.md
└── requirements.txt
```

Файлы `*.csv` и `*.pkl` намеренно исключены из Git. Их необходимо хранить отдельно и передавать защищенным каналом.

## Источник данных

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). Лицензия: CC BY 4.0.

