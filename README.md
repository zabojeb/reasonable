# Reasonable

**Reasonable** — это универсальная библиотека для улучшения рассуждений (reasoning) больших языковых моделей (LLM) без дополнительного дообучения. С её помощью можно интегрировать современные техники, такие как **Chain-of-Thought (CoT)**, **Tree-of-Thought (ToT)**, **Self-Consistency** и **ReAct (Reasoning + Acting)**, в любой проект. Библиотека поддерживает работу с любыми моделями, от облачных (OpenAI, Anthropic, Google Gemini) до локальных (LLaMA, Falcon, Mistral и др.) через текстовый API.

## Основные возможности

- **Chain-of-Thought**: итеративное пошаговое рассуждение, позволяющее модели генерировать цепочку мыслей перед формированием финального ответа.
- **Tree-of-Thought**: разветвлённое построение цепочки размышлений, позволяющее моделям генерировать несколько альтернативных путей решения и выбирать наилучший.
- **Self-Consistency**: запуск нескольких независимых прогонов рассуждений с последующим голосованием за наиболее частый (и, как правило, правильный) ответ.
- **ReAct**: комбинация рассуждений с возможностью вызова внешних инструментов (например, калькулятора, поиска), что повышает точность ответов на задачи, требующие дополнительных данных или вычислений.
- **Оптимизация производительности**: гибкие настройки параметров (максимальное число шагов, ветвление, количество прогонов) и оптимизированные промпты, позволяющие сбалансировать точность и скорость.
- **Логирование и визуализация**: сохранение всех шагов рассуждений в виде дерева или списка, возможность экспорта логов в JSON/CSV, а также встроенная функция текстовой визуализации дерева мыслей.

## Установка

Клонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/zabojeb/reasonable.git
cd reasonable
pip install -r requirements.txt
```

## Примеры использования

### Chain-of-Thought (Базовый агент)

```python
from reasonable import DefaultReasoningAgent

def model_func(prompt: str) -> str:
    # Здесь должна быть реализация вызова вашей LLM, например, OpenAI API.
    return "Пример ответа модели с шагом размышления..."

agent = DefaultReasoningAgent(model_func=model_func, max_steps=5, verbose=True)

result = agent.reason("Какова столица Франции?")

print("Финальный ответ:", result["answer"])
print("Цепочка мыслей:", result["thoughts"])
```

### Tree-of-Thought

```python
from reasonable import TreeReasoningAgent

# model_func определена как функция, вызывающая вашу LLM.
tree_agent = TreeReasoningAgent(model_func=model_func, max_steps=3, branch_factor=2)
result = tree_agent.reason("Вычислите факториал 5 (5!)")
print("Финальный ответ:", result["answer"])
tree_agent.visualize_tree()
tree_agent.save_log("tree_log.json", format="json")
```

### Self-Consistency

```python
from reasonable import SelfConsistencyAgent

sc_agent = SelfConsistencyAgent(model_func=model_func, num_runs=3, max_steps=4)
result = sc_agent.reason("Сколько будет 2 + 2 * 2?")
print("Наиболее частый ответ:", result["answer"])
sc_agent.save_log("sc_log.csv", format="csv")
```

### ReAct (Reasoning + Acting)

```python
from reasonable import ReActAgent

# Определяем инструменты, например, калькулятор и поисковую функцию
def Calc(expression: str) -> str:
    import math
    try:
        return str(eval(expression, {"__builtins__": None, "math": math}))
    except Exception as e:
        return f"Ошибка: {e}"

def Search(query: str) -> str:
    # Фиктивная реализация поиска (можно подключить реальное API)
    if "столица Франции" in query.lower():
        return "Париж"
    return "Результатов не найдено"

tools = {"Calc": Calc, "Search": Search}

react_agent = ReActAgent(model_func=model_func, tools=tools, max_steps=5)
result = react_agent.reason("Вычислите площадь круга радиусом 5. Используйте формулу Pi * r^2.")
print("Финальный ответ:", result["answer"])
react_agent.save_log("react_log.json", format="json")
```

## Вклад в исследования

Проект Reasonable опирается на последние исследования в области улучшения рассуждений LLM, в том числе:
- **Chain-of-Thought Prompting** – метод, позволяющий моделям решать сложные задачи путем пошагового рассуждения.
- **Self-Consistency** – генерация нескольких цепочек рассуждений и выбор наиболее согласованного ответа.
- **Tree-of-Thought** – построение разветвленного дерева мыслей для исследования альтернативных решений.
- **ReAct** – объединение рассуждений и действий для повышения точности при необходимости внешних данных.

Например, статья [Deep Reasoning in Language Models (arXiv:2503.01307)](https://arxiv.org/abs/2503.01307) демонстрирует, как объединение этих подходов может значительно повысить надежность и точность моделей. Наш проект интегрирует эти передовые методики в удобный фреймворк, который можно легко адаптировать под любые задачи, требующие логического мышления и объяснимости решений.

## Лицензия

Этот проект распространяется под лицензией GPL v3. Подробности смотрите в файле [LICENSE](LICENSE).
