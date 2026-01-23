import os
import tempfile
import glob
import pormake as pm
from pymatgen.core import Structure


class BuildCOF2D:
    def __init__(self, ild_guess: float = 15.0):
        self.ild_guess = ild_guess

    def _topology_paths(self, topo: str):
        base = os.path.join(os.path.dirname(pm.__file__), "database", "topologies")
        pickle_path = os.path.join(base, f"{topo}_modified.pickle")
        cgd_path = os.path.join(base, f"{topo}_modified.cgd")
        return pickle_path, cgd_path

    def update_cgd_file(self, new_value: float, topo: str):
        pickle_path, cgd_path = self._topology_paths(topo)

        if os.path.exists(pickle_path):
            os.remove(pickle_path)

        with open(cgd_path, "r") as file:
            lines = file.readlines()

        line_parts = lines[3].split()
        line_parts[3] = f"{float(new_value):.5f}"
        lines[3] = "  ".join(line_parts) + "\n"

        with open(cgd_path, "w") as file:
            file.writelines(lines)

    def extract_cell_lengths(self, cif_file: str):
        structure = Structure.from_file(cif_file)
        return structure.lattice.a, structure.lattice.b, structure.lattice.c

    def calculate_gamma(self, a: float, topo: str):
        alpha = 1.73205 if topo == "hcb" else 1.0
        return (alpha * self.ild_guess) / a

    def _list_xyz(self, folder: str):
        files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in files]

    def build(
        self,
        topo: str,
        nodes_dir: str = "nodes",
        linker_dir: str = "linker",
        output_folder: str = "cofs_test",
        nodename: str | None = None,
        linkername: str | None = None,
    ):
        database = pm.Database()
        topo_initial = database.get_topo(f"{topo}_initial")

        if nodename:
            nodes = [(nodename, os.path.join(nodes_dir, f"{nodename}.xyz"))]
        else:
            nodes = self._list_xyz(nodes_dir)

        if linkername:
            linkers = [(linkername, os.path.join(linker_dir, f"{linkername}.xyz"))]
        else:
            linkers = self._list_xyz(linker_dir)

        builder = pm.Builder()
        edgetype_raw = topo_initial.edge_types
        filtered_edge = edgetype_raw[(edgetype_raw != -1).any(axis=1)]
        unique_pairs = set()
        edgetype_filtered = []
        for pair in filtered_edge:
            if tuple(pair) not in unique_pairs:
                unique_pairs.add(tuple(pair))
                edgetype_filtered.append(tuple(pair))

        os.makedirs(output_folder, exist_ok=True)
        outputs = []

        for node_name, node_path in nodes:
            for linker_name, linker_path in linkers:
                node = pm.BuildingBlock(node_path)
                linker = pm.BuildingBlock(linker_path)

                node_bbs = {0: node}
                edge_bbs = {pair: linker for pair in edgetype_filtered}

                cof = builder.build_by_type(
                    topology=topo_initial,
                    node_bbs=node_bbs,
                    edge_bbs=edge_bbs,
                )

                with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                    tmp_path = tmp.name
                cof.write_cif(tmp_path)
                a, b, c = self.extract_cell_lengths(tmp_path)
                os.remove(tmp_path)

                gamma = self.calculate_gamma(a, topo)
                self.update_cgd_file(gamma, topo)

                topo_modified = database.get_topo(f"{topo}_modified")
                cof = builder.build_by_type(
                    topology=topo_modified,
                    node_bbs=node_bbs,
                    edge_bbs=edge_bbs,
                )

                output_filename = os.path.join(output_folder, f"{node_name}_{linker_name}.cif")
                cof.write_cif(output_filename)
                outputs.append(output_filename)

        return outputs


def build_cof_2d(
    topo: str,
    nodes_dir: str = "nodes",
    linker_dir: str = "linker",
    output_folder: str = "cofs_test",
    nodename: str | None = None,
    linkername: str | None = None,
    ild_guess: float = 15.0,
):
    return BuildCOF2D(ild_guess=ild_guess).build(
        topo=topo,
        nodes_dir=nodes_dir,
        linker_dir=linker_dir,
        output_folder=output_folder,
        nodename=nodename,
        linkername=linkername,
    )
