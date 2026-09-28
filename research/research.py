# %% [markdown]
# # Лабораторная работа №2. Эксперименты по настройке модели
# **Лукожев Ильяс Хусенович, А-02м-25**
#
# Цель — освоить пайплайны, генерацию и отбор признаков, настройку RandomForest
# через Optuna и логирование в MLflow 2.16.0. Выполняются обязательные пункты.
# Пункты 11, 13 и 15 со звёздочкой пропущены по выбору студента.
#
# Перед выполнением запустите в отдельном терминале `sh mlflow/start_mlflow.sh`.
# Критерий выбора победителя — средний F1 пяти стратифицированных CV-разбиений
# обучающей части. Тестовые 25% не участвуют в подборе параметров.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "research" / "experiments.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import display
from research.experiments import (
    load_data, configure_mlflow, make_pipeline, train_and_log, select_expert_columns,
    tune_best, comparison_plot, train_production, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, EXPERT_FEATURES, MODEL_NAME, ROOT,
)

client, experiment_id = configure_mlflow()
print("Эксперимент:", experiment_id)
print("MLflow UI: http://127.0.0.1:5000")

# %% [markdown]
# ## 1. Загрузка очищенной выборки и разбиение 75% / 25%
# Используется `data/clean_dataset.pkl` из ЛР1. Категории приводятся к строкам,
# числовые признаки — к float64, чтобы будущий сервис принимал JSON.
# Шесть пропусков были заполнены по всему исходному набору ещё в ЛР1.
# Это наследуемое ограничение: при полноценном переобучении импутацию нужно
# поместить в train-only pipeline. Все новые преобразования в ЛР2 обучаются
# внутри train/CV. Для train_test_split заданы stratify=y и random_state=42.

# %%
X, y, X_train, X_test, y_train, y_test, dataset_hash = load_data()
print("Числовые признаки:", NUMERIC_FEATURES)
print("Категориальные признаки:", CATEGORICAL_FEATURES)
print("Цель: target (0 — нет заболевания, 1 — есть заболевание)")
print(f"Полная выборка: {len(X)}; train: {len(X_train)}; test: {len(X_test)}")
display(pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()}))

# %% [markdown]
# ## 2. Baseline: StandardScaler + TargetEncoder + RandomForest
# `ColumnTransformer` масштабирует числовые признаки и кодирует категории через
# `sklearn.preprocessing.TargetEncoder`. Его fit_transform использует внутренний
# cross-fitting. Полный pipeline заново обучается в каждом CV-разбиении.
# Baseline: 100 деревьев, стандартные параметры, random_state=42.
# ROC-AUC считается по вероятностям класса 1. Модель, параметры, метрики,
# сигнатура, пример входа, зависимости и списки столбцов логируются в MLflow
# с flavour sklearn. Регистрируется первая версия модели.

# %%
results, fitted_models = [], {}
baseline_result, baseline_model, baseline_version = train_and_log(
    "baseline", X_train, y_train, X_test, y_test, dataset_hash,
    run_name="baseline", register=True,
)
results.append(baseline_result)
fitted_models["baseline"] = baseline_model
display(pd.DataFrame([baseline_result]))
display(baseline_model)
print("Версия baseline:", baseline_version)

# %% [markdown]
# ## 3. Feature extraction средствами sklearn
# Для age, thalach, oldpeak добавляем PolynomialFeatures(degree=2) с последующим
# StandardScaler. Временных признаков нет: для этих же трёх столбцов применяем
# KBinsDiscretizer (4 квантильных интервала, ordinal) и масштабирование.
# В итоге 14 исходных преобразованных + 9 polynomial + 3 bins = 26 столбцов.
# Демонстрационный fit_transform не переиспользуется при оценке: каждый pipeline
# обучает собственные преобразования. Имена всех 26 столбцов сохраняются в файл
# и логируются вместе с моделью в features/transformed_columns.json.

# %%
X_train_fe_sklearn, engineered_names, selected_indices = select_expert_columns(X_train, y_train)
print("Размер X_train_fe_sklearn:", X_train_fe_sklearn.shape)
display(pd.DataFrame({"index": range(len(engineered_names)), "feature": engineered_names}))
engineered_result, engineered_model, _ = train_and_log(
    "engineered", X_train, y_train, X_test, y_test, dataset_hash,
    run_name="sklearn_features",
)
results.append(engineered_result)
fitted_models["sklearn_features"] = engineered_model
display(pd.DataFrame([engineered_result]))

# %% [markdown]
# ## 4. Feature selection: экспертный отбор 13 из 26 признаков
# Выбор зафиксирован по EDA ЛР1 до просмотра результатов ЛР2: age, thalach,
# oldpeak, ca; sex, cp, exang, slope, thal; два взаимодействия и два бина.
# Убираем слабые по EDA chol, trestbps, fbs, restecg, линейные дубли, квадраты
# и age_group. Взаимодействия позволяют проверить совместный эффект признаков.
# Это учебная гипотеза: качество может как улучшиться, так и ухудшиться.
# 13 / 26 = 50% соответствует диапазону 20–70%. Между transform и classification
# добавляем selection по индексам. Имена и индексы сохраняются отдельными файлами.

# %%
print("Выбранные имена:", EXPERT_FEATURES)
print("Выбранные индексы:", selected_indices)
selected_result, selected_model, _ = train_and_log(
    "expert_selected", X_train, y_train, X_test, y_test, dataset_hash,
    selected_indices=selected_indices, run_name="expert_selection",
)
results.append(selected_result)
fitted_models["expert_selection"] = selected_model
display(pd.DataFrame([selected_result]))

# %% [markdown]
# ## 5. Optuna: максимизация F1, минимум 10 trials
# Из трёх вариантов выбирается лучший по train CV F1. Тестовые метрики не
# управляют поиском. Выполняем 12 trials с seed=42: n_estimators ∈ {50,100,150,200},
# max_depth ∈ [3,12], max_features ∈ [0.1,1.0]. Явно указано direction="maximize".
# Каждое испытание логируется отдельным вложенным run. После поиска модель
# обучается на 227 строках, проверяется на 76 тестовых и регистрируется в реестре.

# %%
best_before_optuna = max(results, key=lambda result: result["cv_f1_mean"])
print("Лучший набор до Optuna:", best_before_optuna["name"])
study, best_cv_scores, search_run_id = tune_best(
    best_before_optuna["kind"], X_train, y_train, selected_indices, n_trials=12,
)
print("Выполнено trials:", len(study.trials))
print("Лучшее среднее CV F1:", study.best_value)
print("Параметры:", study.best_params)
display(study.trials_dataframe())

# %%
tuned_result, tuned_model, tuned_version = train_and_log(
    best_before_optuna["kind"], X_train, y_train, X_test, y_test, dataset_hash,
    selected_indices=selected_indices, params=study.best_params, run_name="optuna_best",
    cv_scores=best_cv_scores, register=True,
)
results.append(tuned_result)
fitted_models["optuna_best"] = tuned_model
display(pd.DataFrame([tuned_result]))
print("Зарегистрированная версия:", tuned_version)

# %% [markdown]
# ## 6. Сравнение всех моделей
# Таблица содержит обязательные тестовые precision, recall, F1 и ROC-AUC,
# средний train CV F1 и его стандартное отклонение. Если Optuna уступает
# исходному варианту, победителем остаётся конфигурация с более высоким CV F1.

# %%
comparison = comparison_plot(results)
display(comparison)
best_result = max(results, key=lambda result: result["cv_f1_mean"])
print("Победитель по CV:", best_result["name"])
print("Средний CV F1:", best_result["cv_f1_mean"])
print("Тестовый F1 исходной модели:", best_result["f1"])

# %% [markdown]
# ## 7. Финальная Production-модель
# Победитель переобучается на всех 303 строках, получает следующую версию,
# тег Production=true, тег role=Production и alias Production.
# На финальной модели метрики не измеряются: отложенной выборки уже нет.
# Логируются сигнатура, пример входа, requirements и входные/итоговые столбцы.
# MLmodel копируется в research. Проверяются загрузка через реестр и совпадение
# прогнозов после сериализации. Эти артефакты понадобятся в следующей работе.

# %%
production_model, production_metadata = train_production(
    best_result, fitted_models[best_result["name"]], X, y, dataset_hash, client,
)
display(pd.Series(production_metadata).to_frame("value"))
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
display(pd.DataFrame([
    {"version": v.version, "run_id": v.run_id, "status": v.status, "tags": v.tags}
    for v in versions
]))

# %% [markdown]
# ## 8. Выводы и контрольные вопросы
# Итоговая конфигурация и точные результаты приведены в таблице выше,
# research/production.json и README. Каждый вариант оценён одним и тем же
# разбиением (227 / 76) и одинаковыми пятью CV-folds.
#
# В сохранённом прогоне baseline имеет лучший тестовый F1=0.8493 и ROC-AUC=0.9397.
# Добавление sklearn-признаков снижает тестовый F1 до 0.8333, экспертный отбор —
# до 0.8108. Однако train CV F1 после отбора выше: 0.7909 против 0.7675 baseline.
# Критерий выбора зафиксирован заранее по train CV, поэтому Production —
# экспертный отбор. Optuna даёт CV F1=0.7879 и test F1=0.8219: исходный вариант
# не превзойдён по целевому критерию. При CV std=0.0717 небольшие различия
# нельзя считать убедительным доказательством превосходства одного подхода.
#
# - Feature extraction создаёт представления, например взаимодействия и бины,
#   чтобы модель использовала дополнительные закономерности.
# - Feature selection сокращает набор, снижая избыточность и сложность.
# - MLflow хранит параметры, метрики, артефакты, сигнатуры и версии моделей;
#   UI сравнивает эксперименты, реестр определяет версию для сервиса.
# - Гиперпараметры задаются до обучения: число деревьев, глубина, max_features.
#   Optuna выбирает их по CV F1, целевая метрика максимизируется.
# - Pipeline объединяет transform → selection → classification; при необходимости
#   в него включают импутацию или балансировку.
#
# Ограничения: всего 303 объекта; CV std показывает разброс оценок. Тестовые
# сравнения учебные, импутация унаследована от ЛР1. Метрики исходной модели
# нельзя приписывать переобученной на всех данных версии. Для следующей работы
# сохраните всю папку mlflow с базой mlruns.db и mlartifacts.
