from pathlib import Path

import pytest

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_both_modules_share_the_same_120_symbol_crypto_universe():
    settings = Settings(
        momentum_symbols_file=str(ROOT / "config" / "momentum_symbols.txt"),
        tpr_rb_symbols_file=str(ROOT / "config" / "tpr_rb_symbols.txt"),
    )
    mex = settings.momentum_symbol_list
    all_symbols = settings.symbol_list
    tpr_rb = settings.tpr_rb_symbol_list
    mex_core = settings.mex_core_symbol_list

    assert len(all_symbols) == 120
    assert len(set(all_symbols)) == 120
    assert len(mex) == 120
    assert set(mex) == set(all_symbols)
    assert len(mex_core) == 60
    assert len(set(mex_core)) == 60
    assert len(tpr_rb) == 60
    assert len(set(tpr_rb)) == 60
    assert set(mex_core).isdisjoint(tpr_rb)
    excluded_non_crypto = {"NVDABUSD", "TSLABUSD", "SPYBUSD", "AAPLXUSD", "XAUTUSD", "PAXGUSD", "SLVONUSD", "OILUSD"}
    assert set(all_symbols).isdisjoint(excluded_non_crypto)


def test_symbol_universe_files_contain_delta_product_symbols():
    for path in (
        ROOT / "config" / "momentum_symbols.txt",
        ROOT / "config" / "tpr_rb_symbols.txt",
    ):
        symbols = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert all(symbol.endswith("USD") for symbol in symbols)
        assert all(symbol == symbol.upper() for symbol in symbols)
        assert len(symbols) == len(set(symbols)) == 60


def test_fixed_universe_rejects_wrong_symbol_count(tmp_path):
    momentum_path = tmp_path / "momentum_symbols.txt"
    momentum_path.write_text("\n".join(f"TEST{i:03d}USD" for i in range(59)), encoding="utf-8")
    settings = Settings(momentum_symbols_file=str(momentum_path))
    with pytest.raises(ValueError, match="exactly 60 unique symbols"):
        _ = settings.mex_core_symbol_list


def test_fixed_universe_rejects_duplicate_symbols(tmp_path):
    momentum_path = tmp_path / "momentum_symbols.txt"
    symbols = [f"TEST{i:03d}USD" for i in range(59)] + ["TEST000USD"]
    momentum_path.write_text("\n".join(symbols), encoding="utf-8")
    settings = Settings(momentum_symbols_file=str(momentum_path))
    with pytest.raises(ValueError, match="Duplicate symbol"):
        _ = settings.mex_core_symbol_list




def test_legacy_scanner_symbol_limit_cannot_reduce_the_full_universe():
    settings = Settings(
        scanner_symbol_limit=40,  # Legacy setting is ignored; coverage stays complete.
        momentum_symbols_file=str(ROOT / "config" / "momentum_symbols.txt"),
        tpr_rb_symbols_file=str(ROOT / "config" / "tpr_rb_symbols.txt"),
    )

    assert len(settings.symbol_list) == 120
    assert len(settings.momentum_symbol_list) == 120
