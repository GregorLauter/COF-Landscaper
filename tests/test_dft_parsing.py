from pathlib import Path

import pandas as pd
import pytest

import coflandscaper as cl


@pytest.mark.unit
def test_guess_symbol_handles_common_label_patterns() -> None:
    """This test ensures atom labels are normalized to valid element symbols."""
    assert cl.guess_symbol("C1") == "C"
    assert cl.guess_symbol("si2") == "Si"
    assert cl.guess_symbol("cl_12") == "Cl"
    assert cl.guess_symbol("123") is None


@pytest.mark.unit
def test_parse_cell_reads_values_and_strips_uncertainty() -> None:
    """This test ensures CIF cell fields are parsed even with uncertainty notation."""
    text = """
_cell_length_a  10.0(2)
_cell_length_b  11.5
_cell_length_c  22.100(5)
_cell_angle_alpha 90
_cell_angle_beta  90.0
_cell_angle_gamma 120.0(1)
"""
    cell = cl.parse_cell(text)
    assert cell == {
        "a": 10.0,
        "b": 11.5,
        "c": 22.1,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
    }


@pytest.mark.unit
def test_parse_cell_raises_when_parameter_missing() -> None:
    """This test ensures missing required CIF cell fields fail loudly."""
    text = "_cell_length_a 10\n_cell_length_b 10\n"
    with pytest.raises(ValueError, match="Missing cell parameter"):
        cl.parse_cell(text)


@pytest.mark.unit
def test_extract_atoms_reads_fractional_loop_block() -> None:
    """This test ensures atom loops with fractional coordinates are extracted correctly."""
    lines = [
        "data_test",
        "loop_",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "C1 0.1 0.2 0.3",
        "N2 0.4 0.5 0.6",
        "",
    ]
    atoms = cl.extract_atoms(lines)
    assert atoms[0][0] == 6
    assert atoms[0][1:] == (0.1, 0.2, 0.3)
    assert atoms[1][0] == 7
    assert atoms[1][1:] == (0.4, 0.5, 0.6)


@pytest.mark.unit
def test_extract_atoms_raises_when_no_fractional_loop_exists() -> None:
    """This test ensures invalid CIF content without fractional loops is rejected."""
    lines = ["data_test", "loop_", "_atom_site_label", "C1"]
    with pytest.raises(ValueError, match="No atom site loop"):
        cl.extract_atoms(lines)


@pytest.mark.unit
def test_parse_z_L_from_stem_parses_decitags() -> None:
    """This test ensures z and L tags in file stems are converted to decimal shifts."""
    z, l_value = cl.parse_z_L_from_stem("cof_z015_L230")
    assert z == 1.5
    assert l_value == 23.0


@pytest.mark.unit
def test_crystal_sp_extract_energy_requires_completion_and_parses_last_value() -> (
    None
):
    """This test ensures Crystal SP energy parsing uses converged outputs only."""
    obj = cl.CrystalSP()
    text = (
        "random line\n"
        "TOTAL ENERGY + DISP + GCP (AU)   -100.123\n"
        "TOTAL ENERGY + DISP + GCP (AU)   -101.456\n"
        " TELAPSE  0:00:01\n"
        "footer"
    )
    assert obj._extract_energy_au(text) == -101.456

    not_done_text = "TOTAL ENERGY + DISP + GCP (AU) -10.0\nnot finished\n"
    assert obj._extract_energy_au(not_done_text) is None


@pytest.mark.unit
def test_crystal_sp_read_writes_sorted_csv_with_relative_energy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures Crystal SP read writes deterministic CSV rows and relative energies."""
    in_dir = tmp_path / "cof-x" / "2_cof-x_matrix" / "dft_serr"
    s1 = in_dir / "cof-z10_L20"
    s2 = in_dir / "cof-z20_L10"
    s1.mkdir(parents=True)
    s2.mkdir(parents=True)

    s1_out = s1 / "cof-z10_L20.out"
    s2_out = s2 / "cof-z20_L10.out"
    tail = "\n TELAPSE  0:00:01\nfinal"
    s1_out.write_text(
        "TOTAL ENERGY + DISP + GCP (AU) -100.0" + tail,
        encoding="utf-8",
    )
    s2_out.write_text(
        "TOTAL ENERGY + DISP + GCP (AU) -101.0" + tail,
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    csv_path = cl.CrystalSP().read(str(in_dir))
    df = pd.read_csv(csv_path)

    assert list(df.columns) == [
        "structure",
        "z",
        "L",
        "energy_eV",
        "energy_rel_eV",
    ]
    assert list(df["structure"]) == ["cof-z10_L20", "cof-z20_L10"]
    assert pytest.approx(float(df["energy_rel_eV"].min()), abs=1e-12) == 0.0


def test_find_last_occurrence_returns_latest_match_index() -> None:
    """This test ensures keyword scanning returns the last matching line index."""
    lines = ["A", "B keyword", "C", "D keyword"]
    assert cl.find_last_occurrence(lines, "keyword") == 3
    assert cl.find_last_occurrence(lines, "missing") == -1


@pytest.mark.unit
def test_parse_atom_lines_skips_invalid_rows_then_collects_atoms() -> None:
    """This test ensures atom parsing ignores noise and stops cleanly after parsed atoms."""
    lines = [
        "junk line",
        "1 2 C 0.1 0.2 0.3",
        "1 2 O 0.4 0.5 0.6",
        "1 2 ZZ 0.7 0.8 0.9",
        "tail",
    ]
    symbols, coords = cl.parse_atom_lines(lines, start_idx=0)
    assert symbols == ["C", "O"]
    assert coords == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.unit
def test_crystal_generate_input_validates_mode_and_routes_subfolders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures Crystal generate_input validates mode and routes expected input and output folders."""
    calls: list[tuple[str, str]] = []

    def fake_run(
        _self: cl.Crystal, input_folder: str, output_folder: str | None = None
    ) -> None:
        assert output_folder is not None
        calls.append((input_folder, output_folder))

    monkeypatch.setattr(cl.Crystal, "run", fake_run)

    crystal = cl.Crystal(post_block="")
    crystal.generate_input("cof-a", mode="both")
    assert calls == [
        ("cof-a/2_cof-a_matrix/serr", "cof-a/2_cof-a_matrix/dft_serr"),
        ("cof-a/2_cof-a_matrix/incl", "cof-a/2_cof-a_matrix/dft_incl"),
    ]

    with pytest.raises(ValueError, match="mode must be"):
        crystal.generate_input("cof-a", mode="bad")


@pytest.mark.unit
def test_crystal_sp_read_output_appends_dft_suffix_to_default_csv_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures CrystalSP read_output writes _dft-suffixed CSV names by default."""
    calls: list[tuple[str, str | None, str]] = []

    def fake_read(
        _self: cl.CrystalSP,
        input_folder: str,
        output_csv_dir: str | None = None,
        output_filename_suffix: str = "",
    ) -> Path:
        calls.append((input_folder, output_csv_dir, output_filename_suffix))
        out_dir = output_csv_dir or "cof-a/3_cof-a_landscape"
        mode_tag = Path(input_folder).name.replace("dft_", "")
        return Path(
            f"{out_dir}/cof-a_sp_energies_{mode_tag}{output_filename_suffix}.csv"
        )

    monkeypatch.setattr(cl.CrystalSP, "read", fake_read)

    outputs = cl.CrystalSP().read_output("cof-a", mode="both")

    assert calls == [
        ("cof-a/2_cof-a_matrix/dft_serr", None, "_dft"),
        ("cof-a/2_cof-a_matrix/dft_incl", None, "_dft"),
    ]
    assert outputs == [
        Path("cof-a/3_cof-a_landscape/cof-a_sp_energies_serr_dft.csv"),
        Path("cof-a/3_cof-a_landscape/cof-a_sp_energies_incl_dft.csv"),
    ]


@pytest.mark.unit
def test_crystal_sp_read_does_not_create_default_output_dir_on_missing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures failing CrystalSP reads do not leave empty default output folders."""
    in_dir = tmp_path / "cof-x" / "2_cof-x_matrix" / "dft_serr"
    in_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"No valid \.out files found"):
        cl.CrystalSP().read(str(in_dir))

    assert not (tmp_path / "cof-x" / "3_cof-x_landscape").exists()
