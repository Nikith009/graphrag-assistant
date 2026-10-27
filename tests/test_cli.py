import yaml
from conftest import ROOT

from kgrag.cli import main


def test_offline_index_stats_and_query(tmp_path, capsys):
    cfg = tmp_path / "offline.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "corpus": str(ROOT / "data" / "sample"),
                "index_path": str(tmp_path / "graph.json"),
                "cache_dir": str(tmp_path / "cache"),
                "embedder": {"provider": "hashing"},
            }
        )
    )
    replay = str(ROOT / "data" / "sample-extractions.json")

    assert main(["index", "-c", str(cfg), "--replay", replay]) == 0
    assert (tmp_path / "graph.json").exists()

    assert main(["stats", "-c", str(cfg)]) == 0
    question = "Which ferry companies could be affected by the Fjord EV recall?"
    assert main(["query", "-c", str(cfg), question, "--context-only"]) == 0

    out = capsys.readouterr().out
    assert "'entities': 20" in out
    assert "Kestrel Energy -[SUPPLIES]-> Torvik Marine" in out
    assert "torvik-marine#0" in out
