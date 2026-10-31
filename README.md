# poziomka-gguf

Kod i instrukcja do tworzenia plików GGUF modeli Poziomka (architektura `BailingMoeV2ForCausalLM`), które działają w **zwykłym, prekompilowanym llama.cpp** oraz w standardowym LM Studio, bez własnego kodu i bez dodatkowych flag.

Sprawdzone z [Poziomka-Instruct-2026-10-20](https://huggingface.co/cpral/Poziomka-Instruct-2026-10-20) i oficjalnym wydaniem llama.cpp **b12020** (CPU oraz CUDA).

## Szybki start

```bash
pip install -r llama.cpp/requirements.txt   # po pierwszym uruchomieniu, gdy llama.cpp jest już sklonowane
./make_gguf.sh /sciezka/do/Poziomka-Instruct-2026-10-20 ./gguf /sciezka/do/llama-b12020/llama-quantize
```

Skrypt:

1. klonuje llama.cpp na commicie `0ec4027a2` (= wydanie b12020) i nakłada `llama.cpp-bailingmoe2-spm-tokenizer.patch`,
2. konwertuje model HF do BF16 GGUF,
3. uruchamia `pad_gguf.py` (poprawki opisane niżej),
4. kwantyzuje do `Q8_0 Q6_K Q5_K_M Q4_K_M` (zmienna `QUANTS` zmienia listę).

Patch dotyczy tylko konwersji (Python). Do kwantyzacji i uruchamiania wystarczą oficjalne binarki z [wydań llama.cpp](https://github.com/ggml-org/llama.cpp/releases).

## Co i dlaczego jest poprawiane

### 1. Tokenizer (patch do konwertera)

Tokenizer Poziomki to SentencePiece-BPE wyeksportowany do `tokenizer.json` (`byte_fallback`, spacje jako `▁`), a nie BPE w stylu GPT-2, którego konwerter oczekuje dla BailingMoeV2. Bez patcha konwersja kończy się błędem `BPE pre-tokenizer was not recognized`.

Patch zapisuje słownik jako tokenizer typu `llama` (SPM). Konwerter domyślnie daje wszystkim tokenom ten sam score (-1000), przez co llama.cpp scalał tokeny w złej kolejności. ID tokenów są ułożone dokładnie w kolejności merge'y, więc patch ustawia `score = -id`. Po tej zmianie tokenizacja jest identyczna z HF na 68 testowych tekstach (polskie znaki, emoji, kod, tokeny specjalne).

### 2. 32 aktywnych ekspertów (`pad_gguf.py`)

llama.cpp rezerwuje na graf obliczeń `8 × liczba_tensorów` węzłów (`graph_max_nodes` w `src/llama-context.cpp`). Każda warstwa MoE sumuje wyjścia ekspertów, zużywając ok. 2 węzły na aktywnego eksperta. Przy 32 z 128 ekspertów graf nie mieści się w tym limicie i każdy GGUF zrobiony standardowo kończy się przy tworzeniu kontekstu błędem:

```
ggml_new_object: not enough space in the context's memory pool
GGML_ASSERT(obj_new) failed
```

Ten wzór obowiązuje od lipca 2025, więc problem występuje w każdej wersji llama.cpp obsługującej tę architekturę.

Obejście bez zmian w kodzie: od marca 2026 llama.cpp ładuje opcjonalne tensory `.scale` (wprowadzone dla NVFP4) dla każdej architektury, a graf BailingMoeV2 ich nie używa. `pad_gguf.py` dodaje 48 takich tensorów (`blk.N.ffn_{gate,up,down}[_exps].scale`) o wartości 1.0, łącznie ok. 23 KB. Zwiększa to limit (1776 → 2160 węzłów) bez dokładania operacji do grafu, a wynik modelu pozostaje bez zmian. Nawet gdyby przyszła wersja zaczęła te skale stosować, mnożenie przez 1.0 nic nie zmieni.

Wersje llama.cpp starsze niż marzec 2026 odrzucą te pliki (nieznane tensory), ale i tak nie potrafią uruchomić tego modelu.

Sprawdzone, niedziałające alternatywy:
- dodatkowa zerowa warstwa: zużywa tyle węzłów, ile dodaje do limitu,
- warstwy NextN: są ładowane z `TENSOR_SKIP` i nie wliczają się do limitu.

### 3. Szablon czatu

Model był trenowany na `chat_template.jinja` (v13), a nie na prostszym szablonie z `tokenizer_config.json`, który konwerter bierze domyślnie. `pad_gguf.py` osadza v13 z dwiema zmianami:

- usunięte tagi `{% generation %}`, potrzebne tylko do maskowania lossu w treningu (nie każdy silnik Jinja je obsługuje),
- myślenie zawsze wyłączone: odpowiedź asystenta zawsze zaczyna się od `<think></think>\n`, niezależnie od `enable_thinking`, a `reasoning_content` z historii nie trafia do promptu. Model obsługuje tylko tryb błyskawicznej odpowiedzi.

## Weryfikacja

Wszystko sprawdzone na oficjalnych binarkach b12020:

- BF16, Q8_0, Q6_K, Q5_K_M i Q4_K_M ładują się i generują poprawny tekst na CPU,
- CUDA: pełny i częściowy offload, flash attention włączone i wyłączone, `-ub 2048`, `llama-server --parallel 4`,
- generacja zachłanna zgodna z HF transformers. Jedyna różnica wypada w miejscu, gdzie dwa najlepsze tokeny mają w HF identyczne log-prob (-0.813).

Kwanty K zgłaszają `30 of 270 tensor(s) required fallback quantization`. To normalne: tensory `ffn_down_exps` i `ffn_down_shexp` mają szerokość 320, która nie dzieli się przez 256.

## Zalecane samplery

temperature 0.6, top_p 0.9, min_p 0.05, frequency_penalty 0.15
