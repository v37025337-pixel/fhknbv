# DIGITAL_MIND: подключаемый модуль 0.2.3

Объединённое экспериментальное ядро DIGITAL_MIND + L0 + mechanism genesis v0.28, оформленное как Python-модуль и процесс с JSON-запросами. Программа-помощник может вызывать его для симуляции, логических выводов и исполнения сгенерированных механизмов через L0, а затем использовать результаты в своей работе.

Модуль выполняет собственный код ядра. Он не заменяет внутреннюю модель ChatGPT, не получает доступ к телефону и не подключается к ассистенту автоматически после загрузки на GitHub. В текущем сеансе его можно вызывать через доступную среду исполнения. Для другого приложения интеграция выполняется через Python API или запуск процесса.

## GitHub Codespaces

В репозитории выберите **Code → Codespaces → Create codespace**. Конфигурация `.devcontainer/devcontainer.json` установит пакет и его зависимости. Затем в терминале:

```bash
python examples/use_as_module.py
digital-mind run --steps 96 --report report.json
digital-mind-module
```

Последняя команда ждёт JSON-запросы: по одному на строку. Codespaces можно открыть в браузере телефона. Длительная работа зависит от жизненного цикла Codespace; репозиторий сам по себе не запускает постоянный процесс.

## Python API

```python
from digital_mind_core.tool_module import MindModule

core = MindModule()
report = core.execute({"op": "simulate", "steps": 96})
result = core.execute({"op": "reason", "task": {
    "facts": [["error", "prediction"]],
    "rules": [[[["error", "$x"]], ["needs_review", "$x"]]],
    "queries": [["needs_review", "prediction"]]
}})
assert result["inferences"][0]["value"]

mechanism = core.execute({"op": "run_mechanism", "name": "choose_plan_cost",
                         "inputs": {"history_cost": 2, "current_cost": 9}})
assert mechanism["value"] == 2
```

Каждый `MindModule` владеет одним экземпляром ядра. Повторные вызовы используют то же состояние; новый экземпляр начинает отдельную историю. Сохранение механизмов доступно через `core.mind.save_mechanisms(path)`. Восстановление симуляции и ограничение журнала после внешних логических задач описаны в `CORE_README.md`.

## Процесс с JSON-запросами

```bash
python -m digital_mind_core.tool_module --seed 0
```

Примеры строк ввода:

```json
{"id":1,"op":"simulate","steps":96}
{"id":2,"op":"status"}
{"id":3,"op":"run_mechanism","name":"choose_plan_cost","inputs":{"history_cost":2,"current_cost":9}}
```

Каждая непустая строка получает один ответ вида `{"id":1,"ok":true,"result":...}` или `{"id":1,"ok":false,"error":{"type":...,"message":...}}`. Ошибка запроса не закрывает процесс. Максимальный размер строки — 1 000 000 символов, число шагов за вызов — 1–1024. `id` необязателен. Это простой JSON-lines протокол, не MCP и не JSON-RPC.

Недоступные числовые значения в старых логических и планировочных трассах передаются как `null`. Входные JSON-значения `NaN` и `Infinity` отклоняются.

| Команда | Вход | Результат |
| --- | --- | --- |
| `status` | — | Состояние мира, памяти, L0 и механизмов |
| `simulate` | `steps`, по умолчанию 96 | Симуляция и обновлённый отчёт |
| `reason` | Объект `task`; необязательный `include_state` | Символические выводы ядра |
| `run_mechanism` | `name`, объект `inputs` | Значение и способ исполнения |

На seed 0 после 96 шагов автоматически появляется `choose_plan_cost`. Это локальный арбитр по заданной разработчиком спецификации минимума прогнозируемых стоимостей. Наличие синтеза и исполнения не доказывает рост интеллекта: в прежних сравнительных симуляциях арбитр не изменял выбор действий.

Логический ответ по умолчанию содержит выводы и решение. `"include_state":true` добавляет полный снимок символического состояния для диагностики.


## HTTP API для black-box аудита

HTTP-адаптер использует тот же stateful `MindModule`, что и JSONL-процесс: отдельной копии логики ядра нет, а состояние сохраняется между запросами одного процесса.

Локальный запуск:

```bash
digital-mind-http --host 127.0.0.1 --port 8765
```

В Codespaces, чтобы порт можно было форвардить:

```bash
digital-mind-http --host 0.0.0.0 --port 8765
```

Доступны четыре точки:

```text
POST /kernel/execute
GET  /kernel/status
POST /kernel/research
GET  /kernel/self-diagnostic
```

Примеры:

```bash
curl http://127.0.0.1:8765/kernel/status
curl -X POST http://127.0.0.1:8765/kernel/execute \
  -H 'Content-Type: application/json' \
  -d '{"id":"audit-1","op":"simulate","steps":4}'
curl -X POST http://127.0.0.1:8765/kernel/research \
  -H 'Content-Type: application/json' \
  -d '{"query":"bounded program synthesis","num":5}'
curl http://127.0.0.1:8765/kernel/self-diagnostic
```

`/kernel/research` использует существующий Google adapter и требует `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` либо проверенный Google-backed host callback. `/kernel/self-diagnostic` запускает полный репозиторный аудит, включая тесты, поэтому параллельный второй diagnostic получает HTTP 409.

По умолчанию сервер слушает только `127.0.0.1`. Если в Codespaces сделать forwarded port публичным для внешнего black-box теста, после аудита его следует снова закрыть: `execute` изменяет состояние ядра, а self-diagnostic потребляет вычислительные ресурсы.

## Установка и проверка

Нужны Python 3.10+, NumPy, SciPy:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/verify_legacy.py
python examples/use_as_module.py --report examples/module_demo.json
```

GitHub Actions выполняет эти проверки и сохраняет результаты как скачиваемый артефакт. Файл `examples/verification_summary.json` фиксирует локальные проверки данной сборки. Проверка Actions и запуск контейнера Codespaces считаются выполненными только после реального запуска в GitHub.

Версия 0.2.3 добавляет versioned replay migrations: изменение source SHA или библиотек больше не уничтожает совместимый checkpoint автоматически; совместимость подтверждается воспроизведённым state fingerprint, а несовместимый replay требует явной миграции.

Версия 0.2.2 добавляет stateful HTTP-адаптер для black-box аудита поверх существующего MindModule.\n\nВерсия 0.2.1 добавляет модуль, Codespaces и CI к ядру 0.2.0. Архитектура и ограничения базового ядра описаны в `CORE_README.md`; приведённые там старые результаты относятся к 0.2.0. Старые replay-файлы привязаны к исходному коду и не переносятся между версиями; JSON-механизмы имеют отдельный формат.
