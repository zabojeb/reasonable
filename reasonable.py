from reasonable_prompts import base_reasonable_prompt, final_answer_base_prompt

import re
import time


class DefaultReasoningAgent:
    """
    Default Reasoning Agent that uses CoT (Chain-of-Thought) technique
    to improve answer quality.
    """

    def __init__(
        self,
        main_function: callable,
        thoughts_function: callable,
        max_steps: int = 10,
        timeout: int = 0,
        reasoning_prompt: str = base_reasonable_prompt,
        final_answer_prompt: str = final_answer_base_prompt,
        verbose: bool = False,
    ):
        self.main_function = main_function
        self.thoughts_function = thoughts_function
        self.max_steps = max_steps
        self.verbose = verbose
        self.reasoning_prompt = reasoning_prompt
        self.final_answer_prompt = final_answer_prompt
        self.timeout = timeout

        if self.verbose:
            print("ReasoningAgent initialized. Verbose mode is on.")

    def __parse_text(self, text: str) -> tuple:
        reasoning_match = re.search(
            r'<reasoning>\s*(.*?)\s*</reasoning>', text, re.DOTALL | re.IGNORECASE)
        next_action_match = re.search(
            r'<next_action>\s*(.*?)\s*</next_action>', text, re.DOTALL | re.IGNORECASE)

        reasoning = reasoning_match.group(1) if reasoning_match else text
        next_action = next_action_match.group(
            1) if next_action_match else "continue"

        if self.verbose:
            print(
                f"Parsed from assistant:\n  reasoning: {reasoning}\n  next_action: {next_action}")

        return reasoning, next_action

    def reason(self, user_input) -> dict:
        # For one message
        if isinstance(user_input, str):
            messages = [
                {"role": "system", "content": self.reasoning_prompt},
                {"role": "user", "content": f"<user_request>\n{user_input}\n</user_request>\n<previous_thoughts></previous_thoughts>"},
            ]

            # CoT Loop
            thoughts_array = []
            for step in range(self.max_steps):
                generated = self.thoughts_function(messages)

                thoughts, next_action = self.__parse_text(generated)
                thoughts_array.append(thoughts)

                if "final_answer" in next_action.lower() or step == self.max_steps - 1:
                    break

                stringified_thoughts = "\n".join(thoughts_array)
                messages = [
                    {"role": "system", "content": self.reasoning_prompt},
                    {"role": "user", "content": f"<user_request>\n{user_input}\n</user_request>\n<previous_thoughts>\n{stringified_thoughts}\n</previous_thoughts>"},
                ]

                time.sleep(self.timeout)

            # Final Answer
            stringified_thoughts = "\n".join(thoughts_array)
            messages = [
                {"role": "system", "content": self.final_answer_prompt},
                {"role": "user", "content": f"<user_request>\n{user_input}\n</user_request>\n<previous_thoughts>\n{stringified_thoughts}\n</previous_thoughts>"},
            ]

            time.sleep(self.timeout)

            answer = self.main_function(messages)

            return {"answer": answer, "thoughts": thoughts_array}

        # For message list
        elif isinstance(user_input, list):
            # Начальные сообщения: добавляем системное с промптом для генерации рассуждений
            messages = [
                {"role": "system", "content": self.reasoning_prompt},
                # Далее помещаем всю историю сообщений, которая пришла в user_input
                *user_input,
                # Добавляем "пустой" блок <previous_thoughts> для начала цикла
                {"role": "user", "content": "<previous_thoughts></previous_thoughts>"}
            ]

            thoughts_array = []
            for step in range(self.max_steps):
                # Генерируем очередные размышления с помощью thoughts_function
                generated = self.thoughts_function(messages)

                # Парсим размышления и следующий шаг
                thoughts, next_action = self.__parse_text(generated)
                thoughts_array.append(thoughts)

                # Если модель решила выдать финальный ответ или достигнут максимум шагов — останавливаемся
                if "final_answer" in next_action.lower() or step == self.max_steps - 1:
                    break

                # Обновляем <previous_thoughts> новым контентом
                stringified_thoughts = "\n".join(thoughts_array)

                # Перезаписываем последнее сообщение пользователя, в котором храним цепочку размышлений
                messages[-1] = {
                    "role": "user",
                    "content": f"<previous_thoughts>\n{stringified_thoughts}\n</previous_thoughts>"
                }

                time.sleep(self.timeout)

            # Когда итерации закончены или пришла команда final_answer, формируем запрос на финальный ответ
            stringified_thoughts = "\n".join(thoughts_array)
            final_messages = [
                {"role": "system", "content": self.final_answer_prompt},
                {
                    "role": "user",
                    "content": f"<previous_thoughts>\n{stringified_thoughts}\n</previous_thoughts>"
                },
            ]

            time.sleep(self.timeout)
            answer = self.main_function(final_messages)

            return {"answer": answer, "thoughts": thoughts_array}


class TreeReasoningAgent(DefaultReasoningAgent):
    """Агент, реализующий подход Tree-of-Thought (дерево мыслей) для поиска решения."""

    def __init__(self, model_func, max_steps=5, branch_factor=3, **kwargs):
        """
        model_func: функция для вызова LLM (принимает строку промпта, возвращает строку ответа модели).
        max_steps: максимальная глубина (количество шагов) размышлений.
        branch_factor: количество ветвей (вариантов мыслей) развиваемых на каждом шаге.
        """
        super().__init__(model_func, max_steps=max_steps, **kwargs)
        self.branch_factor = branch_factor
        self.tree = None

    class Node:
        """Вспомогательный класс узла дерева мыслей."""

        def __init__(self, text="", parent=None):
            self.text = text        # Текст данного шага размышления
            self.children = []      # Список дочерних узлов (продолжений мысли)
            self.parent = parent    # Ссылка на родительский узел
            self.final_answer = None  # Если данный узел завершает решение, хранит финальный ответ

        def to_dict(self):
            """Рекурсивно представить узел и его дочерние узлы в виде словаря (для JSON)."""
            result = {"thought": self.text}
            if self.final_answer is not None:
                result["final_answer"] = self.final_answer
            if self.children:
                result["children"] = [child.to_dict()
                                      for child in self.children]
            return result

    def _explore(self, node, question, depth):
        """Рекурсивно развернуть дерево мыслей от данного узла."""
        if depth >= self.max_steps:
            # Достигнута максимальная глубина: получаем финальный ответ для текущей ветви
            prompt_final = self._format_final_prompt(
                question, [n.text for n in self._get_path(node)])
            answer = self.model_func(prompt_final).strip()
            node.final_answer = answer
            return

        # Формируем промпт с текущим состоянием (все мысли от корня до текущего узла) и инструкцией продолжить размышления
        path_thoughts = [n.text for n in self._get_path(node)]
        prompt = self._format_thought_prompt(
            question, path_thoughts, allow_branching=True)

        suggestions = []
        for i in range(self.branch_factor):
            suggestion = self.model_func(prompt).strip()
            if "FINAL_ANSWER" in suggestion:
                answer_text = suggestion.split("FINAL_ANSWER:")[-1].strip()
                node.final_answer = answer_text
                return

            thought_text = suggestion
            child = TreeReasoningAgent.Node(text=thought_text, parent=node)
            node.children.append(child)
            suggestions.append(child)

        for child in suggestions:
            if child.final_answer is None:
                self._explore(child, question, depth+1)

    def _get_path(self, node):
        """Получить список узлов от корня до данного узла."""
        path = []
        cur = node
        while cur is not None:
            path.insert(0, cur)
            cur = cur.parent
        return path

    def reason(self, question):
        """Запуск размышления по методу Tree-of-Thought. Возвращает лучший найденный ответ и лог дерева."""
        self.tree = TreeReasoningAgent.Node(text="Начало размышлений")
        self._explore(self.tree, question, depth=0)

        final_answer = None
        if self.tree.children:
            answers = []

            def collect_answers(node):
                if node.final_answer:
                    answers.append(node.final_answer)
                for ch in node.children:
                    collect_answers(ch)
            collect_answers(self.tree)
            if answers:
                final_answer = answers[0]

        if not final_answer:
            final_answer = "(Не удалось найти уверенный ответ)"

        return {"thoughts_tree": self.tree.to_dict(), "answer": final_answer}

    def save_log(self, filepath, format="json"):
        """Сохранить лог всего дерева размышлений в файл (JSON или CSV)."""
        import json
        import csv
        if format == "json":
            data = self.tree.to_dict() if self.tree else {}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["level", "thought", "final_answer"])

                def write_node(node, level=0):
                    writer.writerow(
                        [level, node.text, node.final_answer or ""])
                    for child in node.children:
                        write_node(child, level+1)
                if self.tree:
                    write_node(self.tree)
        else:
            raise ValueError(
                "Unsupported format for log. Use 'json' or 'csv'.")

    def visualize_tree(self):
        """Вывести дерево размышлений в читаемом текстовом формате."""
        def print_node(node, indent=0):
            prefix = "    " * indent + ("- " if indent > 0 else "")
            text = node.text
            if node.final_answer:
                text += f" -> [Ответ: {node.final_answer}]"
            print(prefix + text)
            for child in node.children:
                print_node(child, indent+1)
        if self.tree:
            print_node(self.tree)
        else:
            print("Дерево размышлений пусто.")


class SelfConsistencyAgent(DefaultReasoningAgent):
    """Агент, реализующий метод Self-Consistency."""

    def __init__(self, model_func, base_agent_class=None, num_runs=5, **kwargs):
        """
        model_func: функция для вызова LLM.
        base_agent_class: класс базового агента.
        num_runs: количество независимых запусков цепочки размышлений.
        """
        super().__init__(model_func, **kwargs)
        self.num_runs = num_runs
        self.base_agent_class = base_agent_class
        self.runs_results = []

    def reason(self, question):
        """Запустить несколько цепочек размышлений и вернуть наиболее согласованный ответ."""
        self.runs_results = []
        answers = []
        for i in range(self.num_runs):
            if self.base_agent_class:
                base_agent = self.base_agent_class(
                    self.model_func, max_steps=self.max_steps)
                result = base_agent.reason(question)
            else:
                result = super().reason(question)

            self.runs_results.append(result)
            answers.append(result["answer"])

        best_answer = None
        if answers:
            from collections import Counter
            counts = Counter(answers)

            best_answer = counts.most_common(1)[0][0]
        else:
            best_answer = "(нет ответа)"

        best_run_thoughts = None
        for res in self.runs_results:
            if res["answer"] == best_answer:
                best_run_thoughts = res.get(
                    "thoughts") or res.get("thoughts_tree") or res
                break

        return {"thoughts": best_run_thoughts, "answer": best_answer}

    def save_log(self, filepath, format="json"):
        """Сохранить логи всех запусков reasoning в файл JSON/CSV."""
        import json
        import csv
        if format == "json":
            data = {"runs": self.runs_results}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["run_index", "answer"])
                for i, res in enumerate(self.runs_results):
                    writer.writerow([i+1, res["answer"]])
        else:
            raise ValueError("Unsupported format for log.")
