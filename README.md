# Квантование Gemma3-12B с I-Matrix (llama.cpp)

Этот README описывает полный пайплайн подготовки калибровочного текста из научных PDF-статей и квантования модели **Gemma3-12B** в формат **GGUF** с использованием **матрицы важности (I-Matrix)** из `llama.cpp` (`llama-imatrix`) и последующего квантования через `llama-quantize`.

Цель: получить более качественное квантование (лучше удержание качества) за счёт калибровки на доменном тексте (научные статьи).

**1. Подготовка PDF → TXT**

1.1. Подобрать научные статьи
Подойдут статьи, близкие к целевому домену использования модели, например:

* ML / NLP / CV (arXiv)

* математика и оптимизация

* физика / инженерия

* обзоры (survey papers)

* sections: Introduction / Methods / Experiments / Discussion

Рекомендации:

* лучше брать много статей среднего объёма, чем 1–2 очень большие

* не использовать “мусорные” источники (презентации вместо статей)
  
1.2. Переконвертировать pdf в txt с помощью python-скрипта или стороннего сервиса

1.3. Нормализовать и очистить txt

После извлечения в тексте обычно встречаются:

* переносы слов и строк

* повторяющиеся пробелы

* битые символы

* “мусор” вроде номеров страниц, колонтитулов и артефактов верстки

* служебные символы

Всё это нужно постараться убрать.

1.4. Собрать единый калибровочный файл

**2. Сборка llama.cpp**

2.1. Клонируем репозиторий:

`git clone https://github.com/ggml-org/llama.cpp.git`

`cd llama.cpp`

2.2. Сборка (CPU):

`cmake -B build -DCMAKE_BUILD_TYPE=Release`

`cmake --build build -j`

Сборка с поддержкой CUDA (NVIDIA GPU)

Убедитесь, что установлен CUDA Toolkit и nvidia-smi работает.

`cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON`

`cmake --build build-cuda -j`

Также можно скачать уже готовый release.

После сборки появятся бинарники:

build/bin/llama-imatrix

build/bin/llama-quantize

build/bin/llama-cli

**3. Скачивание модели**

Скачать базовую модель с https://huggingface.co/.

Если модели в формате GGUF нет, можно воспользоваться конвертацией из safetensors с помощью python

`cd llama.cpp`

`python3 convert_hf_to_gguf.py \
  /absolute/path/to/models/gemma3-12b \
  --outtype f16 \
  --outfile /absolute/path/to/models/gemma3-12b-f16.gguf`

**4. Создание I-Matrix (llama-imatrix)**

Матрица важности строится на основе прогонов модели по калибровочному тексту.
Это позволяет llama-quantize более аккуратно распределять ошибку квантования по весам.

4.1. Запуск llama-imatrix

Пример команды:

`cd llama.cpp`

`./build/bin/llama-imatrix \
  -m /absolute/path/to/models/gemma3-12b-f16.gguf \
  -f /absolute/path/to/data/calib.txt \
  -o /absolute/path/to/models/imatrix.gguf \
  -ngl 0 \
  -c 4096 \
  --chunks 256`

Параметры:

* -m — базовая GGUF модель (обычно F16)

* -f — калибровочный текст

* -o — файл матрицы важности

* -c 4096 — контекст (подбирайте под модель и память)

* --chunks — количество чанков текста (чем больше, тем дольше, но стабильнее)

* -ngl 0 — режим CPU (если есть GPU-сборка, можно ускорить)

**5. Квантование модели с I-Matrix (llama-quantize)**

Теперь квантование выполняется с учётом imatrix.gguf.

5.1. Пример квантования в Q4_K_M

`cd llama.cpp`

`./build/bin/llama-quantize \
  --imatrix /absolute/path/to/models/imatrix.dat \
  /absolute/path/to/models/gemma3-12b-f16.gguf \
  /absolute/path/to/models/gemma3-12b-q4_k_m.gguf \
  Q4_K_M`

Где:

* --imatrix — матрица важности

* входной файл — F16 GGUF

* выходной файл — квантованный GGUF

* Q4_K_M — тип квантования

Другие типы квантования можно посмотреть тут https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py#L2905

**6. Проверка результата**

6.1. Простой запуск инференса

`cd llama.cpp`

`./build/bin/llama-cli \
  -m /absolute/path/to/models/gemma3-12b-q4_k_m.gguf \
  -p "Explain the main idea of self-attention in Transformers." \
  -n 200 \
  -c 4096 \
  --temp 0.7`

6.2. Сравнение качества (F16 vs Quant)

Рекомендуется прогнать одинаковые промпты на:

gemma3-12b-f16.gguf

gemma3-12b-q4_k_m.gguf

и сравнить:

* связность ответа

* точность терминов

* удержание научного стиля

* склонность к “галлюцинациям”
