# Module adapted from https://github.com/nschloe/meshio
# Corrections have been made for Fluent mixed meshes

"""
I/O for Ansys's msh format.

<https://romeo.univ-reims.fr/documents/fluent/tgrid/ug/appb.pdf>
"""

import re
import numpy as np
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


def _skip_to(f, char):
    c = None
    while c != char:
        c = f.read(1).decode()


def _skip_close(f, num_open_brackets):
    while num_open_brackets > 0:
        char = f.read(1).decode()
        if char == "(":
            num_open_brackets += 1
        elif char == ")":
            num_open_brackets -= 1


def _read_points(f, line, first_point_index_overall, last_point_index):
    # If the line is self-contained, it is merely a declaration
    # of the total number of points.
    if line.count("(") == line.count(")"):
        return None, None, None

    # (3010 (zone-id first-index last-index type ND)
    out = re.match("\\s*\\(\\s*(|20|30)10\\s*\\(([^\\)]*)\\).*", line)
    assert out is not None
    a = [int(num, 16) for num in out.group(2).split()]

    if len(a) <= 4:
        raise ReadError()

    first_point_index = a[1]
    # store the very first point index
    if first_point_index_overall is None:
        first_point_index_overall = first_point_index
    # make sure that point arrays are subsequent
    if last_point_index is not None:
        if last_point_index + 1 != first_point_index:
            raise ReadError()
    last_point_index = a[2]
    num_points = last_point_index - first_point_index + 1
    dim = a[4]

    # Skip ahead to the byte that opens the data block (might
    # be the current line already).
    last_char = line.strip()[-1]
    while last_char != "(":
        last_char = f.read(1).decode()

    if out.group(1) == "":
        # ASCII data
        pts = np.empty((num_points, dim))
        for k in range(num_points):
            # skip ahead to the first line with data
            line = ""
            while line.strip() == "":
                line = f.readline().decode()
            dat = line.split()
            if len(dat) != dim:
                raise ReadError()
            for d in range(dim):
                pts[k][d] = float(dat[d])
    else:
        # binary data
        if out.group(1) == "20":
            dtype = np.float32
        else:
            if out.group(1) != "30":
                ReadError(f"Expected keys '20' or '30', got {out.group(1)}.")
            dtype = np.float64
        # read point data
        pts = np.fromfile(f, count=dim * num_points, dtype=dtype).reshape(
            (num_points, dim)
        )

    # make sure that the data set is properly closed
    _skip_close(f, 2)
    return pts, first_point_index_overall, last_point_index


def _read_cells(f, line):
    # If the line is self-contained, it is merely a declaration of the total number of
    # points.
    if line.count("(") == line.count(")"):
        return None, None

    out = re.match("\\s*\\(\\s*(|20|30)12\\s*\\(([^\\)]+)\\).*", line)
    assert out is not None
    a = [int(num, 16) for num in out.group(2).split()]
    if len(a) <= 4:
        raise ReadError()
    first_index = a[1]
    last_index = a[2]
    num_cells = last_index - first_index + 1
    zone_type = a[3]
    element_type = a[4]

    if zone_type == 0:
        # dead zone
        return None, None

    key, num_nodes_per_cell = {
        0: ("mixed", None),
        1: ("triangle", 3),
        2: ("tetra", 4),
        3: ("quad", 4),
        4: ("hexahedron", 8),
        5: ("pyramid", 5),
        6: ("wedge", 6),
    }[element_type]

    # Skip to the opening `(` and make sure that there's no non-whitespace character
    # between the last closing bracket and the `(`.
    if line.strip()[-1] != "(":
        c = None
        while True:
            c = f.read(1).decode()
            if c == "(":
                break
            if not re.match("\\s", c):
                # Found a non-whitespace character before `(`.
                # Assume this is just a declaration line then and
                # skip to the closing bracket.
                _skip_to(f, ")")
                return None, None

    if key == "mixed":
        # From
        # <https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node1470.htm>:
        #
        # > If a zone is of mixed type (element-type=0), it will have a body that
        # > lists the element type of each cell.
        #
        # No idea where the information other than the element types is stored
        # though. Skip for now.
        data = None
    else:
        # read cell data
        if out.group(1) == "":
            # ASCII cells
            data = np.empty((num_cells, num_nodes_per_cell), dtype=int)
            for k in range(num_cells):
                line = f.readline().decode()
                dat = line.split()
                if len(dat) != num_nodes_per_cell:
                    raise ReadError()
                data[k] = [int(d, 16) for d in dat]
        else:
            if key == "mixed":
                raise ReadError("Cannot read mixed cells in binary mode yet")
            # binary cells
            if out.group(1) == "20":
                dtype = np.int32
            else:
                if out.group(1) != "30":
                    ReadError(f"Expected keys '20' or '30', got {out.group(1)}.")
                dtype = np.int64
            shape = (num_cells, num_nodes_per_cell)
            count = shape[0] * shape[1]
            data = np.fromfile(f, count=count, dtype=dtype).reshape(shape)

    # make sure that the data set is properly closed
    _skip_close(f, 2)
    return key, data


def _read_faces(f, line):
    # faces
    # (13 (zone-id first-index last-index type element-type))

    # If the line is self-contained, it is merely a declaration of
    # the total number of points.
    if line.count("(") == line.count(")"):
        return {}

    out = re.match("\\s*\\(\\s*(|20|30)13\\s*\\(([^\\)]+)\\).*", line)
    assert out is not None
    a = [int(num, 16) for num in out.group(2).split()]

    if len(a) <= 4:
        raise ReadError()
    first_index = a[1]
    last_index = a[2]
    num_cells = last_index - first_index + 1
    element_type = a[4]

    element_type_to_key_num_nodes = {
        0: ("mixed", None),
        2: ("line", 2),
        3: ("triangle", 3),
        4: ("quad", 4),
        5: ("pentagon", 5),
        6: ("hexagon", 6),
        7: ("heptagon", 7),
        8: ("octagon", 8),
        9: ("nonagon", 9),
        10: ("decagon", 10),
    }

    # key, num_nodes_per_cell = element_type_to_key_num_nodes[element_type]
    key = "mixed"

    # Skip ahead to the line that opens the data block (might be
    # the current line already).
    if line.strip()[-1] != "(":
        _skip_to(f, "(")

    data = {}
    if out.group(1) == "":
        # ASCII
        if key == "mixed":
            # From
            # <https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node1471.htm>:
            #
            # > If the face zone is of mixed type (element-type = > 0), the body of the
            # > section will include the face type and will appear as follows
            # >
            # > type v0 v1 v2 c0 c1
            # >
            for k in range(num_cells):
                line = ""
                while line.strip() == "":
                    line = f.readline().decode()
                dat = line.split()
                type_index = int(dat[0], 16)
                if type_index == 0:
                    raise ReadError()
                type_string, num_nodes_per_cell = element_type_to_key_num_nodes[
                    type_index
                ]
                if len(dat) != num_nodes_per_cell + 3:
                    raise ReadError()

                if type_string not in data:
                    data[type_string] = []

                data[type_string].append(
                    [int(d, 16) for d in dat[1 : num_nodes_per_cell + 1]]
                )

            data = {key: np.array(data[key]) for key in data}

        else:
            # read cell data
            data = np.empty((num_cells, num_nodes_per_cell), dtype=int)
            for k in range(num_cells):
                line = f.readline().decode()
                dat = line.split()
                # The body of a regular face section contains the grid connectivity, and
                # each line appears as follows:
                #   n0 n1 n2 cr cl
                # where n* are the defining nodes (vertices) of the face, and c* are the
                # adjacent cells.
                if len(dat) != num_nodes_per_cell + 3:
                    raise ReadError()
                data[k] = [int(d, 16) for d in dat[:num_nodes_per_cell]]
            data = {key: data}
    else:
        # binary
        if out.group(1) == "20":
            dtype = np.int32
        else:
            if out.group(1) != "30":
                ReadError(f"Expected keys '20' or '30', got {out.group(1)}.")
            dtype = np.int64

        if key == "mixed":
            raise ReadError("Mixed element type for binary faces not supported yet")

        # Read cell data.
        # The body of a regular face section contains the grid
        # connectivity, and each line appears as follows:
        #   n0 n1 n2 cr cl
        # where n* are the defining nodes (vertices) of the face,
        # and c* are the adjacent cells.
        shape = (num_cells, num_nodes_per_cell + 2)
        count = shape[0] * shape[1]
        data = np.fromfile(f, count=count, dtype=dtype).reshape(shape)
        # Cut off the adjacent cell data.
        data = data[:, :num_nodes_per_cell]
        data = {key: data}

    # make sure that the data set is properly closed
    _skip_close(f, 2)

    return data


def read_fluent_mesh_file(filename):  # noqa: C901

    points = []
    cells = []

    first_point_index_overall = None
    last_point_index = None

    # read file in binary mode since some data might be binary
    with open_file(filename, "rb") as f:
        while True:
            line = f.readline().decode()
            if not line:
                break

            if line.strip() == "":
                continue

            # expect the line to have the form
            #  (<index> [...]
            out = re.match("\\s*\\(\\s*([0-9]+).*", line)
            if not out:
                raise ReadError()
            index = out.group(1)

            if index == "0":
                # Comment.
                _skip_close(f, line.count("(") - line.count(")"))
            elif index == "1":
                # header
                # (1 "<text>")
                _skip_close(f, line.count("(") - line.count(")"))
            elif index == "2":
                # dimensionality
                # (2 3)
                _skip_close(f, line.count("(") - line.count(")"))
            elif re.match("(|20|30)10", index):
                # points
                pts, first_point_index_overall, last_point_index = _read_points(
                    f, line, first_point_index_overall, last_point_index
                )

                if pts is not None:
                    points.append(pts)

            elif re.match("(|20|30)12", index):
                # cells
                # (2012 (zone-id first-index last-index type element-type))
                key, data = _read_cells(f, line)
                if data is not None:
                    cells.append((key, data))

            elif re.match("(|20|30)13", index):
                data = _read_faces(f, line)

                for key in data:
                    cells.append((key, data[key]))

            elif index == "39":  # TODO: implement wall zones
                # warn("Zone specification not supported yet. Skipping.")
                _skip_close(f, line.count("(") - line.count(")"))

            elif index == "45":
                # (45 (2 fluid solid)())
                obj = re.match("\\(45 \\([0-9]+ ([\\S]+) ([\\S]+)\\)\\(\\)\\)", line)
                if obj:
                    warn(
                        f"Zone specification not supported yet ({obj.group(1)}, {obj.group(2)}). "
                        + "Skipping.",
                    )
                else:
                    warn("Zone specification not supported yet.")

            else:
                warn(f"Unknown index {index}. Skipping.")
                # Skipping ahead to the next line with two closing brackets.
                _skip_close(f, line.count("(") - line.count(")"))

    points = np.concatenate(points)

    # Gauge the cells with the first point_index.
    for k, c in enumerate(cells):
        cells[k] = (c[0], c[1] - first_point_index_overall)

    # Convert cells to single list
    cell_list = []
    for cell_type in cells:
        for cell in cell_type[1]:
            cell_list.append(cell.tolist())

    return points, cell_list
