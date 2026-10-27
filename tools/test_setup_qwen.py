"""tools/test_setup_qwen.py - CYBER-FROST-3.8-PS in the setup and the dashboard: its name everywhere ("CYBER-FROST 3.8 ·
Q5_K_M"), the setup ids, its configs recognised (not taken for DeepSeek's --pack), the pinned download manifest, the
context recommended by the VRAM, the drafting choice written into the config, the Models page's settings, and its
template's thinking (always on; thinking off closes the block as its Qwen base writes it).  No GPU, no network.

    python -m unittest tools.test_setup_qwen -v
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402

TEMPLATE = ("{%- set enable_thinking = true %}{%- for m in messages %}"
            "{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
            "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}")


def qwen_cfg(d: Path, mtp: bool = True) -> dict:
    pack = d / "CYBER-FROST-3.8-PS" / "pack"
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    if mtp:
        (pack / "mtp").mkdir()
        (pack / "mtp" / "dense.txt").write_text("x")
    g = str(pack.parent / "CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf")
    return {"exe": "/b/maya", "args": ["--pack", str(pack), "--native", g, "--ple-gguf", g, "--spec", "4", "--prefill",
                                       "auto", "--expert-cache", "auto", "--max-context", "65536"],
            "model_name": "cyber-frost-3.8-ps", "gpu": [0], "tokenizer": str(pack / "tokenizer"),
            "installer": {"family": "qwen", "quant": "CYBER-FROST-3.8-PS", "qwen_quant": "q5_k_m"}}


class Names(unittest.TestCase):
    def test_shown(self):
        self.assertEqual(Q.shown("qwen", "q5_k_m", 112.53), "CYBER-FROST 3.8 · Q5_K_M (113 GB)")
        self.assertEqual(Q.config_stem("qwen", "q5_k_m"), "maya-cyber-frost")

    def test_ids(self):
        for name in ("cyber-frost", "CYBER-FROST-3.8-PS", "cyber-frost-3.8-ps-q5_k_m"):
            self.assertEqual(Q.resolve(name), ("qwen", "q5_k_m"), name)

    def test_family(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = qwen_cfg(Path(t))
            self.assertEqual(Q.family_of(cfg), "qwen")
            cfg.pop("installer")                       # (by its arguments too: --pack with --ple-gguf is not DeepSeek's)
            self.assertEqual(Q.family_of(cfg), "qwen")
            self.assertEqual(Q.config_label(qwen_cfg(Path(t) / "b")), "CYBER-FROST 3.8 · Q5_K_M")

    def test_model_option(self):
        a = type("A", (), {"model": "cyber-frost", "mm3_quant": None})()
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.model, maya.QWEN)


class Setup(unittest.TestCase):
    def test_manifest_pinned(self):
        m = maya.qwen_entry()
        self.assertEqual(m["repo"], "peasantsmith/CYBER-FROST-3.8-PS-GUFF")
        self.assertEqual(len(m["revision"]), 40)
        for name, sha in m["sha256"].items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertGreater(m["files"][name], 100e9)
        self.assertTrue((ROOT / "LICENSE.qwen-community-1.0").is_file())

    def test_context_by_vram(self):
        self.assertEqual(maya.qwen_recommended_context(32), 262144)   # (INT8 K / V: 3.5 GB at 256K)
        self.assertEqual(maya.qwen_recommended_context(16), 131072)
        self.assertEqual(maya.qwen_recommended_context(8), 65536)
        self.assertEqual(maya.qwen_recommended_context(4), 32768)
        self.assertIn(maya.QWEN_DEFAULT_CONTEXT, maya.QWEN_CONTEXTS)

    def test_drafting_chosen(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "maya-cyber-frost.json"
            p.write_text(json.dumps(qwen_cfg(Path(t))))
            with patch.object(maya, "ok"), patch.object(maya, "say"):
                maya.keep_drafting(p, True)
                args = json.loads(p.read_text())["args"]
                self.assertEqual(Path(args[args.index("--mtp") + 1]).name, "mtp")
                self.assertNotIn("--mtp-max-t", args)
                maya.keep_drafting(p, False)
            cfg = json.loads(p.read_text())
            self.assertIn("--mtp", cfg["args"])            # (the server always loads the drafter)
            self.assertEqual(cfg["args"][cfg["args"].index("--mtp-max-t") + 1], "1")
            self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "none")
            self.assertFalse(maya.needs_speed_calibration(cfg))


class Dashboard(unittest.TestCase):
    def test_models_page(self):
        from serve import models as M
        with tempfile.TemporaryDirectory() as t:
            cfg = qwen_cfg(Path(t))
            self.assertEqual(M.choices(cfg), {"draft": ["mtp", "none"], "images": False})
            s = M.settings_of(cfg)
            self.assertEqual((s["draft"], s["kv"], s["expert_cache"]), ("none", "fp16", None))
            out = M.apply(cfg, {"draft": "mtp", "kv": "int8", "ram_gb": 40, "context": 131072})
            self.assertEqual(out["args"][out["args"].index("--kv") + 1], "int8")
            self.assertTrue(out["args"][out["args"].index("--mtp") + 1].endswith("mtp"))
            self.assertNotIn("--mtp-max-t", out["args"])
            off = M.apply(out, {"draft": "none"})
            self.assertIn("--mtp", off["args"])
            self.assertEqual(off["args"][off["args"].index("--mtp-max-t") + 1], "1")
            self.assertEqual(M.settings_of(off)["draft"], "none")
            self.assertEqual(out["env"], {"MAYA_EXPERT_RAM_GB": "40"})
            with self.assertRaises(ValueError):
                M.apply(cfg, {"images": True})
            with self.assertRaises(ValueError):
                M.apply(qwen_cfg(Path(t) / "b", mtp=False), {"draft": "mtp"})

    def test_thinking_always_on_and_off_form(self):
        from serve.frontend import ChatTemplate
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "chat_template.jinja"
            p.write_text(TEMPLATE, encoding="utf-8")
            tpl = ChatTemplate(p)
            self.assertTrue(tpl.forced_thinking)
            msgs = [{"role": "user", "content": "hi"}]
            self.assertTrue(tpl.render(msgs, reasoning_effort="max").endswith("<|im_start|>assistant\n<think>\n"))
            self.assertTrue(tpl.render(msgs, enable_thinking=False).endswith("<think>\n\n</think>\n\n"))


if __name__ == "__main__":
    unittest.main()
