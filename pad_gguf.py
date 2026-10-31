#!/usr/bin/env python3
# Workaround for stock llama.cpp: its graph node budget (8 * n_tensors) is too small for
# BailingMoe2 with 32 active experts (each MoE layer aggregates experts with ~2 nodes per
# expert), so context creation fails with "not enough space in the context's memory pool".
#
# llama.cpp (since 2026-03-11) loads optional per-tensor ".scale" tensors for any arch.
# BailingMoe2 does not use the dense FFN / routed expert scales in its graph, so adding
# them as 1.0 raises n_tensors (and the node budget) at zero graph cost, and stays exact
# if a later version starts applying them.
#
# Also embeds chat_template.jinja (the template the model was trained with), minus the
# training-only {% generation %} tags, with thinking forced off (<think></think>).
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ.get("LLAMA_CPP_DIR", Path(__file__).parent / "llama.cpp")) / "gguf-py"))
import gguf  # noqa: E402

src, dst, template_path = sys.argv[1], sys.argv[2], sys.argv[3]
reader = gguf.GGUFReader(src)
arch = reader.fields["general.architecture"].contents()
assert arch == "bailingmoe2"

template = Path(template_path).read_text(encoding="utf-8")
template = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", template)

# the model only supports instant answers: force the think-off block everywhere and ignore enable_thinking
template_edits = [
    ("{%- if enable_thinking is defined and enable_thinking is not sameas true and enable_thinking is not sameas false -%}\n"
     "    {{- raise_exception('enable_thinking musi być wartością bool.') -}}\n"
     "{%- endif -%}\n", ""),
    ("            {%- if message.get('reasoning_content') -%}\n"
     "                {{- '<think>\\n' + message['reasoning_content'] + '\\n</think>\\n' -}}\n"
     "            {%- elif not (message['content'] or '').startswith('<think>') -%}\n",
     "            {%- if not (message['content'] or '').startswith('<think>') -%}\n"),
    ("    {%- if enable_thinking is sameas true -%}\n"
     "        {{- '<think>\\n' -}}\n"
     "    {%- elif enable_thinking is sameas false -%}\n"
     "        {{- '<think></think>\\n' -}}\n"
     "    {%- endif -%}\n",
     "    {{- '<think></think>\\n' -}}\n"),
]
for old, new in template_edits:
    assert template.count(old) == 1, old
    template = template.replace(old, new)
assert "enable_thinking" not in template.split("#}", 1)[1]

writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

for key, field in reader.fields.items():
    if key.startswith("GGUF.") or key == "general.architecture":
        continue
    val = field.contents()
    if key == "tokenizer.chat_template":
        val = template
    vtype = field.types[0]
    sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(key, val, vtype, sub_type=sub_type)

n_expert = reader.fields[f"{arch}.expert_count"].contents()
names = {t.name for t in reader.tensors}
n_added = 0

for t in reader.tensors:
    # BF16 data comes back as raw bytes, the writer derives the element shape from it
    writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

for name in sorted(names):
    m = re.fullmatch(r"(blk\.\d+\.ffn_(?:gate|up|down)(_exps)?)\.weight", name)
    if m is None:
        continue
    scale_name = f"{m.group(1)}.scale"
    assert scale_name not in names
    shape = (n_expert,) if m.group(2) else (1,)
    writer.add_tensor(scale_name, np.ones(shape, dtype=np.float32))
    n_added += 1

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()
print(f"wrote {dst} (+{n_added} scale tensors)")
