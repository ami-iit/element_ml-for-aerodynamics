"""
I/O SU2 mesh format
<https://su2code.github.io/docs_v7/Mesh-File/>
"""

import copy
import numpy as np
from numpy.typing import ArrayLike
from itertools import chain, islice
from rich.console import Console
from contextlib import contextmanager


def warn(string, highlight: bool = True) -> None:
    Console(stderr=True).print(
        f"[yellow][bold]Warning:[/bold] {string}[/yellow]", highlight=highlight
    )


class ReadError(Exception):
    pass


def is_buffer(obj, mode):
    return ("r" in mode and hasattr(obj, "read")) or (
        "w" in mode and hasattr(obj, "write")
    )


@contextmanager
def open_file(path_or_buf, mode="r"):
    if is_buffer(path_or_buf, mode):
        yield path_or_buf
    else:
        with open(path_or_buf, mode) as f:
            yield f


topological_dimension = {
    "line": 1,
    "polygon": 2,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "hexahedron": 3,
    "wedge": 3,
    "pyramid": 3,
    "line3": 1,
    "triangle6": 2,
    "quad9": 2,
    "tetra10": 3,
    "hexahedron27": 3,
    "wedge18": 3,
    "pyramid14": 3,
    "vertex": 0,
    "quad8": 2,
    "hexahedron20": 3,
    "triangle10": 2,
    "triangle15": 2,
    "triangle21": 2,
    "line4": 1,
    "line5": 1,
    "line6": 1,
    "tetra20": 3,
    "tetra35": 3,
    "tetra56": 3,
    "quad16": 2,
    "quad25": 2,
    "quad36": 2,
    "triangle28": 2,
    "triangle36": 2,
    "triangle45": 2,
    "triangle55": 2,
    "triangle66": 2,
    "quad49": 2,
    "quad64": 2,
    "quad81": 2,
    "quad100": 2,
    "quad121": 2,
    "line7": 1,
    "line8": 1,
    "line9": 1,
    "line10": 1,
    "line11": 1,
    "tetra84": 3,
    "tetra120": 3,
    "tetra165": 3,
    "tetra220": 3,
    "tetra286": 3,
    "wedge40": 3,
    "wedge75": 3,
    "hexahedron64": 3,
    "hexahedron125": 3,
    "hexahedron216": 3,
    "hexahedron343": 3,
    "hexahedron512": 3,
    "hexahedron729": 3,
    "hexahedron1000": 3,
    "wedge126": 3,
    "wedge196": 3,
    "wedge288": 3,
    "wedge405": 3,
    "wedge550": 3,
    "VTK_LAGRANGE_CURVE": 1,
    "VTK_LAGRANGE_TRIANGLE": 2,
    "VTK_LAGRANGE_QUADRILATERAL": 2,
    "VTK_LAGRANGE_TETRAHEDRON": 3,
    "VTK_LAGRANGE_HEXAHEDRON": 3,
    "VTK_LAGRANGE_WEDGE": 3,
    "VTK_LAGRANGE_PYRAMID": 3,
}

num_nodes_per_cell = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "quad8": 8,
    "tetra": 4,
    "hexahedron": 8,
    "hexahedron20": 20,
    "hexahedron24": 24,
    "wedge": 6,
    "pyramid": 5,
    #
    "line3": 3,
    "triangle6": 6,
    "quad9": 9,
    "tetra10": 10,
    "hexahedron27": 27,
    "wedge15": 15,
    "wedge18": 18,
    "pyramid13": 13,
    "pyramid14": 14,
    #
    "line4": 4,
    "triangle10": 10,
    "quad16": 16,
    "tetra20": 20,
    "wedge40": 40,
    "hexahedron64": 64,
    #
    "line5": 5,
    "triangle15": 15,
    "quad25": 25,
    "tetra35": 35,
    "wedge75": 75,
    "hexahedron125": 125,
    #
    "line6": 6,
    "triangle21": 21,
    "quad36": 36,
    "tetra56": 56,
    "wedge126": 126,
    "hexahedron216": 216,
    #
    "line7": 7,
    "triangle28": 28,
    "quad49": 49,
    "tetra84": 84,
    "wedge196": 196,
    "hexahedron343": 343,
    #
    "line8": 8,
    "triangle36": 36,
    "quad64": 64,
    "tetra120": 120,
    "wedge288": 288,
    "hexahedron512": 512,
    #
    "line9": 9,
    "triangle45": 45,
    "quad81": 81,
    "tetra165": 165,
    "wedge405": 405,
    "hexahedron729": 729,
    #
    "line10": 10,
    "triangle55": 55,
    "quad100": 100,
    "tetra220": 220,
    "wedge550": 550,
    "hexahedron1000": 1000,
    "hexahedron1331": 1331,
    #
    "line11": 11,
    "triangle66": 66,
    "quad121": 121,
    "tetra286": 286,
}


class CellBlock:
    def __init__(
        self,
        cell_type: str,
        data: list | np.ndarray,
        tags: list[str] | None = None,
    ):
        self.type = cell_type
        self.data = data

        if cell_type.startswith("polyhedron"):
            self.dim = 3
        else:
            self.data = np.asarray(self.data)
            self.dim = topological_dimension[cell_type]

        self.tags = [] if tags is None else tags

    def __repr__(self):
        items = [
            "meshio CellBlock",
            f"type: {self.type}",
            f"num cells: {len(self.data)}",
            f"tags: {self.tags}",
        ]
        return "<" + ", ".join(items) + ">"

    def __len__(self):
        return len(self.data)


class Mesh:
    def __init__(
        self,
        points: ArrayLike,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
        point_data: dict[str, ArrayLike] | None = None,
        cell_data: dict[str, list[ArrayLike]] | None = None,
        field_data=None,
        point_sets: dict[str, ArrayLike] | None = None,
        cell_sets: dict[str, list[ArrayLike]] | None = None,
        gmsh_periodic=None,
        info=None,
    ):
        self.points = np.asarray(points)
        if isinstance(cells, dict):
            # Let's not deprecate this for now.
            # warn(
            #     "cell dictionaries are deprecated, use list of tuples, e.g., "
            #     '[("triangle", [[0, 1, 2], ...])]',
            #     DeprecationWarning,
            # )
            # old dict, deprecated
            #
            # convert dict to list of tuples
            cells = list(cells.items())

        self.cells = []
        for cell_block in cells:
            if isinstance(cell_block, tuple):
                cell_type, data = cell_block
                cell_block = CellBlock(
                    cell_type,
                    # polyhedron data cannot be converted to numpy arrays
                    # because the sublists don't all have the same length
                    data if cell_type.startswith("polyhedron") else np.asarray(data),
                )
            self.cells.append(cell_block)

        self.point_data = {} if point_data is None else point_data
        self.cell_data = {} if cell_data is None else cell_data
        self.field_data = {} if field_data is None else field_data
        self.point_sets = {} if point_sets is None else point_sets
        self.cell_sets = {} if cell_sets is None else cell_sets
        self.gmsh_periodic = gmsh_periodic
        self.info = info

        # assert point data consistency and convert to numpy arrays
        for key, item in self.point_data.items():
            self.point_data[key] = np.asarray(item)
            if len(self.point_data[key]) != len(self.points):
                raise ValueError(
                    f"len(points) = {len(self.points)}, "
                    f'but len(point_data["{key}"]) = {len(self.point_data[key])}'
                )

        # assert cell data consistency and convert to numpy arrays
        for key, data in self.cell_data.items():
            if len(data) != len(cells):
                raise ValueError(
                    f"Incompatible cell data '{key}'. "
                    f"{len(cells)} cell blocks, but '{key}' has {len(data)} blocks."
                )

            for k in range(len(data)):
                data[k] = np.asarray(data[k])
                if len(data[k]) != len(self.cells[k]):
                    raise ValueError(
                        "Incompatible cell data. "
                        + f"Cell block {k} ('{self.cells[k].type}') "
                        + f"has length {len(self.cells[k])}, but "
                        + f"corresponding cell data item has length {len(data[k])}."
                    )

    def __repr__(self):
        lines = ["<meshio mesh object>", f"  Number of points: {len(self.points)}"]
        special_cells = [
            "polygon",
            "polyhedron",
            "VTK_LAGRANGE_CURVE",
            "VTK_LAGRANGE_TRIANGLE",
            "VTK_LAGRANGE_QUADRILATERAL",
            "VTK_LAGRANGE_TETRAHEDRON",
            "VTK_LAGRANGE_HEXAHEDRON",
            "VTK_LAGRANGE_WEDGE",
            "VTK_LAGRANGE_PYRAMID",
        ]
        if len(self.cells) > 0:
            lines.append("  Number of cells:")
            for cell_block in self.cells:
                string = cell_block.type
                if cell_block.type in special_cells:
                    string += f"({cell_block.data.shape[1]})"
                lines.append(f"    {string}: {len(cell_block)}")
        else:
            lines.append("  No cells.")

        if self.point_sets:
            names = ", ".join(self.point_sets.keys())
            lines.append(f"  Point sets: {names}")

        if self.cell_sets:
            names = ", ".join(self.cell_sets.keys())
            lines.append(f"  Cell sets: {names}")

        if self.point_data:
            names = ", ".join(self.point_data.keys())
            lines.append(f"  Point data: {names}")

        if self.cell_data:
            names = ", ".join(self.cell_data.keys())
            lines.append(f"  Cell data: {names}")

        if self.field_data:
            names = ", ".join(self.field_data.keys())
            lines.append(f"  Field data: {names}")

        return "\n".join(lines)

    def copy(self):
        return copy.deepcopy(self)

    def get_cells_type(self, cell_type: str):
        if not any(c.type == cell_type for c in self.cells):
            return np.empty((0, num_nodes_per_cell[cell_type]), dtype=int)
        return np.concatenate([c.data for c in self.cells if c.type == cell_type])

    def get_cell_data(self, name: str, cell_type: str):
        return np.concatenate(
            [d for c, d in zip(self.cells, self.cell_data[name]) if c.type == cell_type]
        )

    @property
    def cells_dict(self):
        cells_dict = {}
        for cell_block in self.cells:
            if cell_block.type not in cells_dict:
                cells_dict[cell_block.type] = []
            cells_dict[cell_block.type].append(cell_block.data)
        # concatenate
        for key, value in cells_dict.items():
            cells_dict[key] = np.concatenate(value)
        return cells_dict

    @property
    def cell_data_dict(self):
        cell_data_dict = {}
        for key, value_list in self.cell_data.items():
            cell_data_dict[key] = {}
            for value, cell_block in zip(value_list, self.cells):
                if cell_block.type not in cell_data_dict[key]:
                    cell_data_dict[key][cell_block.type] = []
                cell_data_dict[key][cell_block.type].append(value)

            for cell_type, val in cell_data_dict[key].items():
                cell_data_dict[key][cell_type] = np.concatenate(val)
        return cell_data_dict

    @property
    def cell_sets_dict(self):
        sets_dict = {}
        for key, member_list in self.cell_sets.items():
            sets_dict[key] = {}
            offsets = {}
            for members, cells in zip(member_list, self.cells):
                if members is None:
                    continue
                if cells.type in offsets:
                    offset = offsets[cells.type]
                    offsets[cells.type] += cells.data.shape[0]
                else:
                    offset = 0
                    offsets[cells.type] = cells.data.shape[0]
                if cells.type in sets_dict[key]:
                    sets_dict[key][cells.type].append(members + offset)
                else:
                    sets_dict[key][cells.type] = [members + offset]
        return {
            key: {
                cell_type: np.concatenate(members)
                for cell_type, members in sets.items()
                if sum(map(np.size, members))
            }
            for key, sets in sets_dict.items()
        }

    def cell_sets_to_data(self, data_name: str | None = None):
        # If possible, convert cell sets to integer cell data. This is possible if all
        # cells appear exactly in one group.
        default_value = -1
        if len(self.cell_sets) > 0:
            intfun = []
            for k, c in enumerate(zip(*self.cell_sets.values())):
                # Go for -1 as the default value. (NaN is not int.)
                arr = np.full(len(self.cells[k]), default_value, dtype=int)
                for i, cc in enumerate(c):
                    if cc is None:
                        continue
                    arr[cc] = i
                intfun.append(arr)

            for item in intfun:
                num_default = np.sum(item == default_value)
                if num_default > 0:
                    warn(
                        f"{num_default} cells are not part of any cell set. "
                        f"Using default value {default_value}."
                    )
                    break

            if data_name is None:
                data_name = "-".join(self.cell_sets.keys())
            self.cell_data[data_name] = intfun
            self.cell_sets = {}

    def point_sets_to_data(self, join_char: str = "-") -> None:
        # now for the point sets
        # Go for -1 as the default value. (NaN is not int.)
        default_value = -1
        if len(self.point_sets) > 0:
            intfun = np.full(len(self.points), default_value, dtype=int)
            for i, cc in enumerate(self.point_sets.values()):
                intfun[cc] = i

            if np.any(intfun == default_value):
                warn(
                    "Not all points are part of a point set. "
                    f"Using default value {default_value}."
                )

            data_name = join_char.join(self.point_sets.keys())
            self.point_data[data_name] = intfun
            self.point_sets = {}

    # This used to be int_data_to_sets(), converting _all_ cell and point data.
    # This is not useful in many cases, as one usually only wants one
    # particular data array (e.g., "MaterialIDs") converted to sets.
    def cell_data_to_sets(self, key: str):
        """Convert point_data to cell_sets."""
        data = self.cell_data[key]

        # handle all int and uint data
        if not all(v.dtype.kind in ["i", "u"] for v in data):
            raise RuntimeError(f"cell_data['{key}'] is not int data.")

        tags = np.unique(np.concatenate(data))

        # try and get the names by splitting the key along "-" (this is how
        # sets_to_int_data() forms the key)
        names = key.split("-")
        # remove duplicates and preserve order
        # <https://stackoverflow.com/a/7961390/353337>:
        names = list(dict.fromkeys(names))
        if len(names) != len(tags):
            # alternative names
            names = [f"set-{key}-{tag}" for tag in tags]

        # TODO there's probably a better way besides np.where, something from
        # np.unique or np.sort
        for name, tag in zip(names, tags):
            self.cell_sets[name] = [np.where(d == tag)[0] for d in data]

        # remove the cell data
        del self.cell_data[key]

    def point_data_to_sets(self, key: str):
        """Convert point_data to point_sets."""
        data = self.point_data[key]

        # handle all int and uint data
        if not all(v.dtype.kind in ["i", "u"] for v in data):
            raise RuntimeError(f"point_data['{key}'] is not int data.")

        tags = np.unique(data)

        # try and get the names by splitting the key along "-" (this is how
        # sets_to_int_data() forms the key
        names = key.split("-")
        # remove duplicates and preserve order
        # <https://stackoverflow.com/a/7961390/353337>:
        names = list(dict.fromkeys(names))
        if len(names) != len(tags):
            # alternative names
            names = [f"set-key-{tag}" for tag in tags]

        # TODO there's probably a better way besides np.where, something from
        # np.unique or np.sort
        for name, tag in zip(names, tags):
            self.point_sets[name] = np.where(data == tag)[0]

        # remove the cell data
        del self.point_data[key]


# follows VTK conventions
su2_type_to_numnodes = {
    3: 2,  # line
    5: 3,  # triangle
    9: 4,  # quad
    10: 4,  # tetra
    12: 8,  # hexahedron
    13: 6,  # wedge
    14: 5,  # pyramid
}
su2_to_meshio_type = {
    3: "line",
    5: "triangle",
    9: "quad",
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}
meshio_to_su2_type = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
    "tetra": 10,
    "hexahedron": 12,
    "wedge": 13,
    "pyramid": 14,
}


def read(filename):

    with open_file(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    cells = []
    cell_data = {"su2:tag": []}

    itype = "i8"
    ftype = "f8"
    dim = 0

    next_tag_id = 0
    expected_nmarkers = 0
    markers_found = 0
    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == "%":
            continue

        try:
            name, rest_of_line = line.split("=")
        except ValueError:
            warn(f"meshio could not parse line\n {line}\n skipping.....")
            continue

        if name == "NDIME":
            dim = int(rest_of_line)
            if dim != 2 and dim != 3:
                raise ReadError(f"Invalid dimension value {line}")

        elif name == "NPOIN":
            # according to documentation rest_of_line should just be a int,
            # and the next block should be just the coordinates of the points
            # However, some file have one or two extra indices not related to the
            # actual coordinates.
            # So lets read the next line to find its actual number of columns
            #
            first_line = f.readline()
            first_line = first_line.split()
            first_line = np.array(first_line, dtype=ftype)

            extra_columns = first_line.shape[0] - dim

            num_verts = int(rest_of_line.split()[0]) - 1
            points = np.fromfile(
                f, count=num_verts * (dim + extra_columns), dtype=ftype, sep=" "
            ).reshape(num_verts, dim + extra_columns)

            # save off any extra info
            if extra_columns > 0:
                first_line = first_line[:-extra_columns]
                points = points[:, :-extra_columns]

            # add the first line we read separately
            points = np.vstack([first_line, points])

        elif name == "NELEM" or name == "MARKER_ELEMS":
            # we cannot? read at once using numpy because we do not know the
            # total size. Read, instead next num_elems as is and re-use the
            # translate_cells function from vtk reader

            num_elems = int(rest_of_line)
            gen = islice(f, num_elems)

            # some files has an extra int column while other not
            # We do not need it so make sure we will skip it
            first_line_str = next(gen)
            first_line = first_line_str.split()
            nnodes = su2_type_to_numnodes[int(first_line[0])]
            has_extra_column = False
            if nnodes + 1 == len(first_line):
                has_extra_column = False
            elif nnodes + 2 == len(first_line):
                has_extra_column = True
            else:
                raise ReadError(f"Invalid number of columns for {name} field")

            # reset generator
            gen = chain([first_line_str], gen)

            cell_array = " ".join([line.rstrip("\n") for line in gen])
            cell_array = np.fromiter(cell_array.split(), dtype=itype)

            cells_, _ = _translate_cells(cell_array, has_extra_column)

            for eltype, data in cells_.items():
                cells.append(CellBlock(eltype, data))
                num_block_elems = len(data)
                if name == "NELEM":
                    cell_data["su2:tag"].append(
                        np.full(num_block_elems, 0, dtype=np.int32)
                    )
                else:
                    tags = np.full(num_block_elems, next_tag_id, dtype=np.int32)
                    cell_data["su2:tag"].append(tags)

        elif name == "NMARK":
            expected_nmarkers = int(rest_of_line)
        elif name == "MARKER_TAG":
            next_tag = rest_of_line
            try:
                next_tag_id = int(next_tag)
            except ValueError:
                next_tag_id += 1
                warn(
                    "meshio does not support tags of string type.\n"
                    f"    Surface tag {rest_of_line} will be replaced by {next_tag_id}"
                )
            markers_found += 1

    if markers_found != expected_nmarkers:
        warn(
            f"expected {expected_nmarkers} markers according to NMARK value "
            f"but found only {markers_found}"
        )

    # merge boundary elements in a single cellblock per cell type
    if dim == 2:
        types = ["line"]
    else:
        types = ["triangle", "quad"]

    indices_to_merge = {}
    for t in types:
        indices_to_merge[t] = []

    for index, cell_block in enumerate(cells):
        if cell_block.type in types:
            indices_to_merge[cell_block.type].append(index)

    cdata = cell_data["su2:tag"]
    for type, indices in indices_to_merge.items():
        if len(indices) > 1:
            cells[indices[0]] = CellBlock(
                type, np.concatenate([cells[i].data for i in indices])
            )
            cdata[indices[0]] = np.concatenate([cdata[i] for i in indices])

    # delete merged blocks
    idelete = []
    for type, indices in indices_to_merge.items():
        idelete += indices[1:]

    for i in sorted(idelete, reverse=True):
        del cells[i]
        del cdata[i]

    cell_data["su2:tag"] = cdata
    return Mesh(points, cells, cell_data=cell_data)


def _translate_cells(data, has_extra_column=False):
    # adapted from _vtk.py
    # Translate input array  into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (vtk cell type, p0, p1, ... ,pk, vtk cell type, p10, p11, ..., p1k, ...

    entry_offset = 1
    if has_extra_column:
        entry_offset += 1

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    types = []
    i = 0
    while i < len(data):
        types.append(data[i])
        i += su2_type_to_numnodes[data[i]] + entry_offset

    types = np.array(types)
    bins = {u: np.where(types == u)[0] for u in np.unique(types)}

    # Deduct offsets from the cell types. This is much faster than manually
    # going through the data array. Slight disadvantage: This doesn't work for
    # cells with a custom number of points.
    numnodes = np.empty(len(types), dtype=int)
    for tpe, idx in bins.items():
        numnodes[idx] = su2_type_to_numnodes[tpe]
    offsets = np.cumsum(numnodes + entry_offset) - (numnodes + entry_offset)

    cells = {}
    cell_data = {}
    for tpe, b in bins.items():
        meshio_type = su2_to_meshio_type[tpe]
        nnodes = su2_type_to_numnodes[tpe]
        indices = np.add.outer(offsets[b], np.arange(1, nnodes + 1))
        cells[meshio_type] = data[indices]

    return cells, cell_data
