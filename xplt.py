# Base packages
from math import inf
import struct
from warnings import warn

# Third-party public packages
import numpy as np

# Same-package modules
from .core import _canonical_face
from .model import Mesh
from . import element

# Specification metadata for each (documented) tag.
#
# `name` := string.  A human-readable name for the tag.  Names may be
# inconsistent from source to source or even within a source.
#
# `leaf` := logical.  True = this tag tags leaf blocks.  False = this
# tag tags branch blocks.
#
# `format := string (optional).  Only applies to leaf tags.  Specifies
# how the leaf block's data chunk should be decoded.  Allowed values:
# 'int' = one or more integers, 'float' = one or more floats, 'str' = a
# string, 'variable' = format is defined in the file's dictionary
# section.
tags_table = {
    4605250: {"name": "FEBio"},  # 0x00464542; not leaf or branch
    16777216: {"name": "root", "leaf": False},  # 0x01000000
    33554432: {"name": "state", "leaf": False},  # 0x02000000
    16842752: {"name": "header", "leaf": False},  # 0x01010000
    16908288: {"name": "dictionary", "leaf": False},  # 0x01020000
    16973824: {"name": "materials", "leaf": False},  # 0x01030000
    # Root/Materials section tags; xplt 1.0 only.  Isomorphic to Parts
    # in xplt 2.0.
    16973825: {"name": "material", "leaf": False},  # 0x01030001
    16973826: {"name": "material id", "leaf": True, "format": "int"},  # 0x01030002
    16973827: {"name": "material name", "leaf": True, "format": "str"},  # 0x01030003
    # Mesh (2.0) or Geometry (1.0) section
    0x01040000: {"name": "mesh", "leaf": False},  # 17039360
    # Header section tags
    16842753: {"name": "version", "leaf": True, "format": "int"},  # 0x01010001
    16842754: {"name": "nodes", "leaf": True, "format": "int"},  # 0x01010002
    16842755: {"name": "max facet nodes", "leaf": True, "format": "int"},  # 0x01010003
    16842756: {"name": "compression", "leaf": True},  # 0x01010004
    0x01010005: {"name": "author", "leaf": True},
    0x01010006: {"name": "software", "leaf": True},
    # Dictionary section tags; all optional
    16912384: {"name": "global variables", "leaf": False},  # 0x01021000
    16916480: {"name": "material variables", "leaf": False},  # 0x01022000
    16920576: {"name": "node variables", "leaf": False},  # 0x01023000
    16924672: {"name": "domain variables", "leaf": False},  # 0x01024000
    16928768: {"name": "surface variables", "leaf": False},  # 0x01025000
    16908289: {"name": "dictionary item", "leaf": False},  # 0x01020001
    # Dictionary item tags
    0x01020002: {
        "name": "item type",  # 16908290,
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    16908291: {
        "name": "item format",  # 0x01020003
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    16908292: {"name": "item name", "leaf": True, "format": "str"},  # 0x01020004
    # Item array size seems to always be 4 zero bytes
    0x01020005: {"name": "item array size", "leaf": True},  # 16908293
    # Item array name appears to be unused
    0x01020006: {"name": "item array name", "leaf": True},  # 16908294
    # Mesh (2.0) or Root/Geometry (1.0) tags
    17043456: {"name": "nodes", "leaf": False},  # 0x01041000
    17047552: {"name": "domains", "leaf": False},  # 0x01042000
    17051648: {"name": "surfaces", "leaf": False},  # 0x01043000
    17055744: {"name": "nodesets", "leaf": False},  # 0x01044000
    # Mesh/Nodes (2.0) or Root/Geometry/Nodes (1.0) tags
    0x01041100: {"name": "node header", "leaf": False},
    # xplt 1.0
    0x01041001: {"name": "node coords", "leaf": True, "format": "float"},  # 17043457
    # xplt 2.0; mixed integer IDs and floating point positions
    0x01041200: {"name": "node coords", "leaf": True, "format": "bytes"},  # 17043968
    # Mesh/Node Header tags (2.0 only)
    0x01041101: {"name": "size", "leaf": True, "format": "int"},
    0x01041102: {"name": "dimensions", "leaf": True, "format": "int"},
    0x01041103: {"name": "name", "leaf": True, "format": "str"},
    # Mesh/Domains (2.0) or Root/Geometry/Domains (1.0) tags
    17047808: {"name": "domain", "leaf": False},  # 0x01042100
    # Mesh/Domains/Domain tags
    17047809: {"name": "domain_header", "leaf": False},  # 0x01042101
    17048064: {"name": "element_list", "leaf": False},  # 0x01042200
    # Mesh/Domains/Domain/Domain Header tags
    17047810: {
        "name": "element_type",  # 0x01042102
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    0x01042103: {  # was material ID in 1.0, but not used for anything
        "name": "part ID",
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    16982276: {"name": "elements", "leaf": True, "format": "int"},  # 0x01032104
    16982277: {"name": "domain name", "leaf": True, "format": "end_str"},  # 0x01032105
    # Mesh/Domains/Domain/Element List tags
    17048065: {"name": "element", "leaf": True, "format": "int"},  # 0x01042201
    # Mesh/Surfaces (2.0) or Root/Geometry/Surfaces (1.0) tags
    17051904: {"name": "surface", "leaf": False},  # 0x01043100
    # Mesh/Surfaces/Surface tags
    17051905: {"name": "surface header", "leaf": False},  # 0x01043101
    17052160: {"name": "facet list", "leaf": False},  # 0x01043200
    # Mesh/Surfaces/Surface/Surface Header tags
    17051906: {
        "name": "surface ID",  # 0x01043102
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    17051907: {"name": "facets", "leaf": True, "format": "int"},  # 0x01043103
    17051908: {"name": "surface name", "leaf": True, "format": "str"},  # 0x01043104
    0x01043105: {"name": "max facet nodes", "leaf": True, "format": "int"},  # 17051909
    # Mesh/Surfaces/Surface/Facet List tags
    17052161: {"name": "facet", "leaf": True, "format": "int"},  # 0x01043201
    # Mesh/Nodesets (2.0) or Root/Geometry/Nodesets (1.0) tags
    17056000: {"name": "nodeset", "leaf": False},  # 0x01044100
    # Mesh/Nodesets/Nodeset tags
    17056001: {"name": "nodeset header", "leaf": False},  # 0x01044101
    17056256: {"name": "node list", "leaf": True, "format": "int"},  # 0x01044200
    # Mesh/Nodesets/Nodeset Header tags
    17056002: {"name": "nodeset ID", "leaf": True, "format": "int"},  # 0x01044102
    17056004: {"name": "nodes", "leaf": True, "format": "int"},  # 0x01044104
    17056003: {"name": "nodeset name", "leaf": True, "format": "str"},  # 0x01044103
    # Mesh/Parts; xplt 2.0 only
    0x01045000: {"name": "parts", "leaf": False},  # 17059840
    0x01045100: {"name": "part", "leaf": False},  # 17060096
    0x01045101: {"name": "part ID", "leaf": True, "format": "int"},  # 17060097
    0x01045102: {"name": "part name", "leaf": True, "format": "str"},  # 17060098
    # Mesh/Plot Objects; xplt 2.0 only
    0x01050000: {"name": "objects", "leaf": False},  # 17104896
    0x01050001: {"name": "ID", "leaf": True, "format": "int"},  # 17104897
    0x01050002: {"name": "name", "leaf": True, "format": "end_str"},  # 17104898
    0x01050003: {"name": "tag", "leaf": True, "format": "bytes"},  # 17104899
    0x01050004: {"name": "pos", "leaf": True, "format": "float"},  # 17104900
    0x01050005: {"name": "rot", "leaf": True, "format": "float"},  # 17104901
    0x01050006: {"name": "data", "leaf": False, "format": "bytes"},  # 17104902
    # Mesh/Plot Objects/Point; xplt 2.0 only
    0x01051000: {"name": "point", "leaf": False},  # 17108992
    0x01051001: {"name": "coords", "leaf": True, "format": "bytes"},  # 17108992
    # Mesh/Plot Objects/Line; xplt 2.0 only
    0x01052000: {"name": "line", "leaf": False},  # 17113088
    0x01052001: {"name": "coords", "leaf": True, "format": "bytes"},  # 17113089
    # State tags
    33619968: {"name": "state header", "leaf": False},  # 0x02010000
    33685504: {"name": "state data", "leaf": False},  # 0x02020000
    # State/State Header tags
    0x02010002: {  # 33619970
        "name": "time",
        "leaf": True,
        "format": "float",
        "singleton": True,
    },
    0x02010003: {  # 33619971
        "name": "status",
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    # State/Mesh State tags (2.0 only)
    0x02030000: {"name": "mesh state", "leaf": False},  # 33751040
    0x02030001: {"name": "element state", "leaf": True, "format": "bytes"},  # 33751041
    # State/State Data tags
    33685760: {"name": "global data", "leaf": False},  # 0x02020100
    33686016: {"name": "material data", "leaf": False},  # 0x02020200
    33686272: {"name": "node data", "leaf": False},  # 0x02020300
    33686528: {"name": "domain data", "leaf": False},  # 0x02020400
    33686784: {"name": "surface data", "leaf": False},  # 0x02020500
    # State/State Data/* tags
    33685505: {"name": "state variable", "leaf": False},  # 0x02020001
    # State/State Data/*/State Variable tags
    33685506: {
        "name": "variable ID",  # 0x02020002
        "leaf": True,
        "format": "int",
        "singleton": True,
    },
    33685507: {"name": "data", "leaf": True, "format": "variable"},  # 0x02020003
    # State/Objects State
    0x02040000: {"name": "objects state", "leaf": False, "format": "bytes"},  # 33816576
}

# TODO: Version 49 changed rigid body force data, again.
SUPPORTED_XPLT_VERSIONS = {2, 5, 48}
# map of xplt version → path for mesh (2.0) or geometry (1.0) block
MESH_PATH = {
    2: "root/mesh",
    4: "root/mesh",
    5: "root/mesh",
    48: "mesh",
    49: "mesh",
    50: "mesh",
}
# map of xplt version → path for parts (2.0) or materials (1.0) block
PARTS_PATH = {
    2: "root/materials",
    5: "root/materials",
    48: "mesh/parts",
    49: "mesh/parts",
}

element_type_from_id = {
    0: element.Hex8,
    1: element.Penta6,
    2: "tet4",
    3: element.Quad4,
    4: element.Tri3,
    5: "truss2",
    6: "hex20",
    7: "tet10",
    8: "tet15",
    9: element.Hex27,
}

item_type_from_id = {
    0: "float",
    1: "vec3f",
    2: "mat3fs",
    3: "mat3fd",
    4: "tens4fs",
    5: "mat3f",
    6: "array_float",
    7: "array_vec3f",
}

# Size of each value type in plotfile data, in bytes
INT_SZ_B = 4
FLOAT_SZ_B = 4
VEC3F_SZ_B = 12
MAT3FS_SZ_B = 24
VALUE_SZ_B = {
    "int": INT_SZ_B,
    "float": FLOAT_SZ_B,
    "vec3f": VEC3F_SZ_B,
    "mat3fs": MAT3FS_SZ_B,
}

value_layout_from_id = {  # Refer to FE_enum.h:326
    0: "node",
    1: "item",
    2: "mult",
    3: "region",
}

MDATA_PATH_BY_DTYPE = {
    "node": "root/dictionary/node variables/dictionary item",
    "surface": "root/dictionary/surface variables/dictionary item",
    "domain": "root/dictionary/domain variables/dictionary item",
    "object": "mesh/objects/point/data",
}
DATA_PATH_BY_DTYPE = {
    "node": "state/state data/node data/state variable",
    "surface": "state/state data/surface data/state variable",
    "domain": "state/state data/domain data/state variable",
    "object": "state/objects state/point",
}
# First key is data type (node, surface, domain, or object); second key is item
# format / layout (node, item, mult, or region).  It's currently unclear if object
# data can have an item format other than "item".
entity_type_from_data_type = {
    "node": {
        "node": {
            "entity type": "node",
            "region selector": None,
            "parent selector": None,
        },
        "item": {
            "entity type": "node",
            "region selector": None,
            "parent selector": None,
        },
    },
    "surface": {
        "node": {
            "entity type": "node",
            "region selector": True,
            "parent selector": None,
        },
        "item": {
            "entity type": "face",
            "region selector": True,
            "parent selector": None,
        },
        "mult": {
            "entity type": "node",
            "region selector": True,
            "parent selector": "face",
        },
        "region": {
            "entity type": "surface",
            "region selector": None,
            "parent selector": None,
        },
    },
    "domain": {
        "node": {
            "entity type": "node",
            "region selector": True,
            "parent selector": None,
        },
        "item": {
            "entity type": "element",
            "region selector": True,
            "parent selector": None,
        },
        "mult": {
            "entity type": "node",
            "region selector": True,
            "parent selector": "element",
        },
        "region": {
            "entity type": "domain",
            "region selector": None,
            "parent selector": None,
        },
    },
    "object": {
        "item": {
            "entity type": "object",
            "region selector": None,
            "parent selector": None,
        }
    },
}

_PARSE_ERROR_GENERIC = "  One of the following is true: (1) the input data is not valid plotfile data, (2) the file format specification has changed, or (3) there is a bug in waffleiron."

_LOOKUP_ERROR_GENERIC = "  Note that nodes and element IDs are 0-indexed, but surface and domain IDs are read from the plotfile verbatim.  FEBio seems to always use 1-indexed surface and domain IDs."


def parse_endianness(data):
    """Return endianness of xplt data."""
    # Parse identifier tag
    if data[:4] == b"BEF\x00":
        endian = "<"
    elif data[:4] == b"\x00FEB":
        endian = ">"
    else:
        msg = (
            "Input data is not valid as an FEBio binary database file.  "
            "The first four bytes are {}, but should be 'BEF\x00' or "
            "'\x00F\EB'."
        )
        raise ValueError(msg.format(data[:4]))

    return endian


def parse_blocks(data, offset=0, store_data=True, max_depth=inf, endian="<"):
    """Parse data as a FEBio binary database block.

    Inputs:

    data := bytes.  The data to be parsed.

    offset := integer.  The parser returns the address of each tag as
    `offset` + (number of bytes in `data` before the start of the tag).
    This is to facilitate recursive use.

    store_data := logical (default True).  If True, load the data of
    leaf blocks into memory and store it under the 'data' key in the
    returned metadata dictionary for the block.  If False, do not load
    the data of leaf blocks into memory.  The returned dictionary of
    metadata for a block will then have no 'data' key.

    endian := '<' (little endian) or '>' (big endian).  Endianness
    numeric values in `data`.

    max_depth := integer or inf.  Blocks will be be parsed `max_depth`
    levels deep.  A minimum of one level is always parsed in the parse
    tree.

    Output:

    list of dictionaries.

    Each dictionary records the metadata of one block.  Each contains
    the following fields:
    - 'name': human-readable tag name,
    - 'tag': tag value in hex8 notation (e.g., 0x00000000),
    - 'type': 'branch', 'leaf', or 'unknown'
    - 'address': `offset` + (number of bytes in `data` before the start
      of the tag),
    - 'size': number of bytes in the block's data chunk,
    - 'data': For a branch block, a list of dictionaries containing
       metadata for child blocks.  For a leaf block, a bytes string
       containing the block's data.

    The dictionaries are listed in the same order as their corresponding
    blocks occur in `data`.

    """
    # Initialize variables used throughout
    i = 0  # index of current byte in `data`
    blocks = []

    # Traverse the data, looking for blocks
    while i < len(data):
        if i + 8 > len(data):
            warn(
                "A branch node's block contains data that does not unpack into a valid child node.  This data will be skipped."
            )
            break
        # Get block id
        b_id = data[i : i + 4]
        # Get block size
        b_size = data[i + 4 : i + 8]
        i_size = struct.unpack(endian + "I", b_size)[0]
        # Get this block's data/children
        if i + i_size > len(data):
            warn(
                "A branch node's block contains data that does not unpack into a valid child node.  This data will be skipped."
            )
            break
        child = data[i + 8 : i + 8 + i_size]
        # Convert ID from bytes to nicer form
        i_id = struct.unpack(endian + "I", b_id)[0]
        s_id = "0x" + format(i_id, "08x")
        # Collect the block's basic metadata.
        block = {
            "name": "unknown",
            "tag": s_id,
            "type": "unknown",
            "address": i + offset,
            "size": i_size,
        }

        # Try looking up specification metadata.  If we fail, treat this
        # block as a leaf and return the basic metadata.
        try:
            tag_metadata = tags_table[i_id]
        except KeyError:
            if store_data:
                block["data"] = child
            blocks.append(block)
            i += 8 + i_size
            continue
        block["name"] = tag_metadata["name"]
        # If this is a leaf block, parse its data.  If it is a branch
        # block, parse its children.
        if tag_metadata["leaf"]:  # is a leaf block
            block["type"] = "leaf"
            if "format" in tag_metadata:
                fmt = tag_metadata["format"]
                if store_data:
                    block["data"] = unpack_block(child, block["tag"], endian)
            else:
                if store_data:
                    block["data"] = child
        else:  # is a branch block
            block["type"] = "branch"
            if max_depth > 1:
                block["data"] = parse_blocks(
                    child,
                    offset=offset + i + 8,
                    endian=endian,
                    store_data=store_data,
                    max_depth=max_depth - 1,
                )
        # Record this block's metadata and move on
        blocks.append(block)
        i += 8 + i_size
    return blocks


def parse_xplt_data(data, **kwargs):
    """Parse data as an FEBio binary database file.

    The parser is robust to unknown tags, but is not currently robust to
    misaligned tags or incomplete tags.

    """
    # Parse FEBio tag
    endian = parse_endianness(data[:4])

    # Parse the rest of the file
    parse_tree = parse_blocks(data[4:], offset=4, endian=endian, **kwargs)

    return parse_tree


def pprint_blocks(f, blocks, parents=tuple()):
    """Pretty-print parsed XPLT blocks.

    `f` must support a `write` method.

    """
    indent = 2 * len(parents)
    for i, block in enumerate(blocks):
        if i == 0:  # First block
            # Write opening braces
            f.write(" " * indent + "[{")
        else:
            # Write the new block's indent & brace
            f.write(" " * (indent + 1) + "{")
        path = parents + (block["name"],)
        f.write(f"'path': '{'/'.join(path)}'")
        # Write address/offset in hex (to match most hex editors)
        f.write(",\n" + " " * (indent + 2) + f"'offset': '{block['address']:x}'")
        for k in ["tag", "type", "size"]:
            f.write(",\n" + " " * (indent + 2) + "'{}': '{}'".format(k, block[k]))

        # Write block data/children
        if "data" in block:
            if block["type"] == "branch":
                f.write(",\n" + " " * (indent + 2) + "'children':\n")
                pprint_blocks(f, block["data"], parents=path)
            else:  # Leaf or unknown
                f.write(",\n" + " " * (indent + 2) + f"'data': {block['data']!r}")

        if i < len(blocks) - 1:  # Before last block
            # Close the braces for the current block and write a
            # continuation comma
            f.write("},\n")
        else:  # Last block
            # Close the braces for the current block
            f.write("}\n")
            # Close the braces for the list of blocks
            f.write(" " * indent + "]")


def unpack_data(data, val_type, endian):
    float_dtype = np.dtype("float32")
    if val_type in ["int", "float"]:  # Numeric cases
        n = int(len(data) / VALUE_SZ_B[val_type])
        if val_type == "int":
            v = struct.unpack(endian + n * "I", data)
        else:  # float
            v = struct.unpack(endian + n * "f", data)
    elif val_type == "vec3f":
        if len(data) % VEC3F_SZ_B != 0:
            raise ValueError("Input data cannot be evenly divided into " "vectors.")
        v = []
        for j in range(0, len(data), 12):
            v.append(
                np.array(
                    struct.unpack(endian + "f" * 3, data[j : j + 12]), dtype=float_dtype
                )
            )
    elif val_type == "mat3fs":
        if len(data) % MAT3FS_SZ_B != 0:
            raise ValueError("Input data cannot be evenly divided into " "tensor.")
        v = []
        for j in range(0, len(data), 24):
            a = struct.unpack(endian + "f" * 6, data[j : j + 24])
            # The FEBio database spec does not document the
            # tensor order, but this should be accurate.
            v.append(
                np.array(
                    [[a[0], a[3], a[5]], [a[3], a[1], a[4]], [a[5], a[4], a[2]]],
                    dtype=float_dtype,
                )
            )
    elif val_type == "str":
        # FEBio uses null-terminated strings.  Strip the null byte and
        # everything after it, and decode from bytes to text.
        v = data[0 : data.find(b"\x00")].decode()
    elif val_type == "end_str":
        # FEBio sometimes uses null-prefixed strings.  Read from the
        # last null byte to the end of the data payload.
        v = data[data.rfind(b"\x00") + 1 :].decode()
    else:
        # Pass through unrecognized values.
        v = data
    return v


def get_state_var_data(state_block, var_mdata):
    """Return state data for a variable."""
    var_idx = var_mdata["index"]
    var_name = var_mdata["name"]
    rtype = var_mdata["region type"]
    if rtype in ("node", "domain", "surface"):
        # Standard state variables are stored in 'state variable' blocks
        p = f"state/state data/{rtype} data/state variable"
        var_blocks = find_all(state_block, p)
        # Compute map from variable ID (0x02020002) to local index.  Note that the
        # variable ID is converted to a 0-indexed value for consistency with the
        # variable metadata.
        idx_from_id = {
            find_one(b["data"], "variable ID")["data"] - 1: i
            for i, b in enumerate(var_blocks)
        }
        var_id = var_mdata["index"]
        var_block = var_blocks[idx_from_id[var_id]]
        data = find_one(var_block["data"], "data")["data"]
    else:  # rtype = "object"
        # Object state variables are stored in special 'objects state' blocks
        var_blocks = find_all(state_block, "state/objects state/point/data")
        data = b"".join(
            [
                block["data"][var_idx]["data"]
                for var_idx, block in zip(var_mdata["index"], var_blocks)
            ]
        )
        # ^ iterate over objects and concatenate their data for the current variable.
        # `var_idx` is the index of the current variable's data block in the list of
        # data blocks for the object to which `var_blocks` belongs.
    return data


def _regions_in_sdata(sdata, endian):
    """Return dict of region ID → rdata start byte in state data."""
    region_idx = {}
    i = 0  # Bytes offset to current region's ID
    while i < len(sdata):
        region_id, sz = struct.unpack(endian + "II", sdata[i : i + 8])
        region_idx[region_id] = i + 8
        i = i + 8 + sz
    return region_idx


def _iter_step_data(step_blocks, var_mdata, endian):
    """Return a iterator over step and region yielding unpacked data."""
    for step_block in step_blocks:
        time = find_one(step_block["data"], "state header/time")["data"]
        raw = get_state_var_data(step_block, var_mdata)
        for region, values in _iter_region_data(raw, var_mdata["type"], endian):
            yield time, region, values


def _iter_region_data(data, type_, endian):
    """Return an iterator over regions yielding unpacked data.

    Iterate over binary data in the (id, size, values)+ format as used
    in `state_data/*/state_variable/data` blocks.

    Yields (region_id, values) for each region in the data.

    """
    i = 0
    while i < len(data):
        region_id, n = struct.unpack(endian + "II", data[i : i + 8])
        i += 8
        values = unpack_data(data[i : i + n], type_, endian)
        i += n
        yield region_id, values


def unpack_block(data, tag, endian):
    """Unpack a block's data given its tag ID."""
    if isinstance(tag, str) and tag.startswith("0x"):
        tag = int(tag, 16)
    fmt = tags_table[tag]["format"]
    value = unpack_data(data, fmt, endian)
    if tags_table[tag].setdefault("singleton", False):
        value = value[0]
    return value


def get_bdata_by_name(blocks, pth):
    """Return a list of the contents of all blocks matching a path."""
    if not isinstance(blocks, list):
        blocks = [blocks]
    names = pth.split("/")
    while len(names) != 0:
        name, names = names[0], names[1:]
        matches = []
        for b in blocks:
            if b["name"] == name:
                if b["type"] == "branch":
                    matches += b["data"]
                elif b["type"] == "leaf" and len(names) == 0:
                    # Leaf blocks can only match the end of the given
                    # name path.
                    matches.append(b["data"])
        blocks = matches
    return matches


def find_all(blocks, pth):
    """Return all blocks that match the given path."""
    if not isinstance(blocks, list):
        blocks = [blocks]
    names = pth.split("/")
    while len(names) != 0:
        matches = []
        for b in blocks:
            if b["name"] == names[0]:
                if b["type"] == "branch":
                    matches.append(b)
                elif b["type"] == "leaf" and len(names) == 1:
                    # Leaf blocks can only match the end of the given
                    # name path.
                    matches.append(b)
        # Descend one level
        if len(names) > 1:
            blocks = []
            for b in matches:
                blocks += b["data"]
        names = names[1:]
    return matches


def find_one(blocks, pth):
    """Return the block that matches the given path.

    `find_one` is used to signal intent that only one match should
    exist.  If zero or more than one match is found, an exception is
    raised.  It is particularly useful when accessing child blocks that
    store attributes of a parent block.

    """
    matches = find_all(blocks, pth)
    if len(matches) > 1:
        raise ValueError(f"More than one block matches {pth}")
    elif len(matches) == 0:
        raise ValueError(f"No block matches {pth}")
    else:
        return matches[0]


def find_first(blocks, pth):
    """Return first block matching the given path."""
    return find_all(blocks, pth)[0]


def domains(blocks, version):
    """Return a dictionary of mesh domains.

    Domain IDs are 1-indexed, both in the plotfile and in the returned
    dictionary.  They should only ever be used as keys.

    Element IDs are 0-indexed because they are sometimes used as
    sequences indices.

    """
    domain_dict = {}
    b_domains = find_all(blocks, f"{MESH_PATH[version]}/domains/domain")
    for i, b_domain in enumerate(b_domains):
        domain_id = i + 1
        elem_type_id = find_one(b_domain, "domain/domain_header/element_type")["data"]
        elem_type = element_type_from_id[elem_type_id]
        material_id = find_one(b_domain, "domain/domain_header/part ID")["data"]
        domain_name = find_all(b_domain, "domain/domain_header/domain name")
        if len(domain_name) == 0:
            # Some older plotfiles don't store domain names
            domain_name == ""
        else:
            domain_name = domain_name[0]["data"]
        element_ids = [
            t[0] - 1 for t in get_bdata_by_name(b_domain, "domain/element_list/element")
        ]
        # ^ the element IDs are 1-indexed in the plotfile even though
        # the node IDs (which we're discarding) are 0-indexed.
        domain_dict[domain_id] = {
            "name": domain_name,
            "element type": elem_type,
            "material ID": material_id,
            "element IDs": element_ids,
        }
    return domain_dict


def surfaces(blocks, version):
    """Return a dictionary of mesh surfaces.

    Surface IDs are 1-indexed, both in the plotfile and in the returned
    dictionary.

    Each facet is represented as its canonical tuple of (zero-indexed)
    node IDs.  That is, the tuple is rotated such that the lowest ID is
    first.

    """
    surface_dict = {}
    b_surfaces = find_all(blocks, f"{MESH_PATH[version]}/surfaces/surface")
    for b_surface in b_surfaces:
        surface_name = find_one(b_surface["data"], "surface header/surface name")[
            "data"
        ]
        surface_id = find_one(b_surface["data"], "surface header/surface ID")["data"]
        facets = [
            _canonical_face(t[2 : 2 + t[1]])
            for t in get_bdata_by_name(b_surface, "surface/facet list/facet")
        ]
        # ^ the element IDs are 1-indexed in the plotfile even though
        # the node IDs (which we're discarding) are 0-indexed.
        surface_dict[surface_id] = {"name": surface_name, "facets": facets}
    return surface_dict


def variable_metadata(blocks):
    """Return a map of (entity type, variable name) → variable metadata"""
    vars_mdata = {}
    # Object variables.  Each object stores its own list of data blocks, both in the
    # metadata and in the state data itself.  Therefore we will see the same variable
    # multiple times in the metadata.  We also need the index of the variable within
    # each object's list, so we need to store a list of indexes.
    b_objects = find_all(blocks, "mesh/objects/point")
    for idx_object, b_object in enumerate(b_objects):
        b_variables = find_all(b_object["data"], "data")
        for idx_variable, b_variable in enumerate(b_variables):
            var_name = find_one(b_variable["data"], "item name")["data"]
            if idx_object == 0:
                # Initialize the metadata dictionary
                var_type = item_type_from_id[
                    find_one(b_variable["data"], "item type")["data"]
                ]
                layout = value_layout_from_id[
                    find_one(b_variable["data"], "item format")["data"]
                ]
                var_mdata = {
                    "name": var_name,
                    "type": var_type,
                    "region type": "object",
                    "index": [idx_variable],
                    "layout": layout,
                }
                regional_mdata = entity_type_from_data_type["object"][layout]
                var_mdata.update(regional_mdata)
            else:
                # We should have already seen this variable.  Or can
                # different objects store different variables?  /fear
                var_mdata = vars_mdata[(var_name, "object")]
                var_mdata["index"].append(idx_variable)
            vars_mdata[(var_name, "object")] = var_mdata
    # Node, surface, and domain variables
    for entity_type in ("node", "surface", "domain"):
        b_variables = find_all(blocks, MDATA_PATH_BY_DTYPE[entity_type])
        for i, b_variable in enumerate(b_variables):
            var_name = find_one(b_variable["data"], "item name")["data"]
            var_type = item_type_from_id[
                find_one(b_variable["data"], "item type")["data"]
            ]
            layout = value_layout_from_id[
                find_one(b_variable["data"], "item format")["data"]
            ]
            var_mdata = {
                "name": var_name,
                "type": var_type,
                "region type": entity_type,
                "index": i,
                "layout": layout,
            }
            regional_mdata = entity_type_from_data_type[entity_type][layout]
            var_mdata.update(regional_mdata)
            vars_mdata[(var_name, entity_type)] = var_mdata
    return vars_mdata


def _get_nodes_for_face(face, mesh):
    return face


def _get_nodes_for_elem_ID(elem_id, mesh):
    return mesh.elements[elem_id].ids


class XpltData:
    """In-memory storage and reading of xplt file data."""

    def __init__(self, data):
        """Initialize XpltData object from xplt bytes data."""
        self.endian = parse_endianness(data[:4])
        self.blocks = parse_xplt_data(data, store_data=True)
        self.version = find_one(self.blocks, "root/header/version")["data"][0]
        if self.version not in SUPPORTED_XPLT_VERSIONS:
            warn(
                f"Xplt version tag = {self.version}, which is not known to be supported by waffleiron.  Proceeding regardless.  Subsequent errors may be due to unaddressed changes in FEBio's xplt format."
            )

        self.regions = {
            "surface": surfaces(self.blocks, self.version),
            "domain": domains(self.blocks, self.version),
        }

        # Compute map: entity → index in region
        self._regional_idx = {}
        # ^ first key is region ID, second key is data layout, third key
        # is entity ID / canonical representation
        #
        mesh = self.mesh()
        node_id_getter = {
            "surface": _get_nodes_for_face,
            "domain": _get_nodes_for_elem_ID,
        }
        for region_type, id_field in zip(
            ("surface", "domain"), ("facets", "element IDs")
        ):
            for region_id, region_mdata in self.regions[region_type].items():
                self._regional_idx.setdefault(region_type, {})
                self._regional_idx[region_type].setdefault(region_id, {})
                self._regional_idx[region_type][region_id].setdefault("item", {})
                self._regional_idx[region_type][region_id].setdefault("mult", {})
                self._regional_idx[region_type][region_id].setdefault("node", {})
                i_mult = 0  # Next available regional index for "mult" layout
                i_node = 0  # Next available regional index for "node" layout
                traversed_nodes = set()
                for idx_entity, entity in enumerate(region_mdata[id_field]):
                    # "item" data layout: one value per face / element
                    d_item = self._regional_idx[region_type][region_id]["item"]
                    d_item[entity] = idx_entity
                    # "mult" data layout: one value per node per entity.
                    # The map's values are a list of (parent entity ID,
                    # regional index) tuples.
                    d_mult = self._regional_idx[region_type][region_id]["mult"]
                    f = node_id_getter[region_type]
                    node_ids = f(entity, mesh)
                    idx_mult = (i_mult + i for i in range(len(node_ids)))
                    i_mult += len(node_ids)
                    for idx, node_id in zip(idx_mult, node_ids):
                        d_mult.setdefault(node_id, [])
                        d_mult[node_id].append((entity, idx))
                    # "node" data layout: one value per node, byzantine ordering.
                    for node_id in node_ids:
                        if node_id in traversed_nodes:
                            continue
                        self._regional_idx[region_type][region_id]["node"][
                            node_id
                        ] = i_node
                        i_node += 1
                    traversed_nodes.update(node_ids)

        # Store step data
        self.step_blocks = find_all(self.blocks, "state")
        # Step times
        self.step_times = []
        for b in self.step_blocks:
            t = get_bdata_by_name(b["data"], "state header/time")[0]
            self.step_times.append(t)
        # Get region metadata for each variable.  We need to unpack additional values
        # that were not unpacked during the initial call to `parse_xplt_data`,
        # so we need the endiannes of the file.  When this module is modified to use
        # arrays instead of bytes, `parse_xplt_data` should be modified to parse the
        # (region ID, size, region data) sections in the step data.
        self.variables = {
            varname: mdata
            for (varname, etype), mdata in variable_metadata(self.blocks).items()
        }
        for var, mdata in self.variables.items():
            if mdata["region type"] == "object":
                # Object data doesn't have region-specific subsets
                continue
            sdata = get_state_var_data(self.step_blocks[0], mdata)
            region_idxs = _regions_in_sdata(sdata, self.endian)
            # regions: tuple of region IDs for which the variable is
            # defined.  This is meant to assist the user in browsing
            # variables.
            mdata["regions"] = tuple(region_idxs.keys())
            # _region_idx: dictionary mapping region ID → byte offset
            # of its region data within the state data.   This is to
            # enable other functions to look up values by region.
            mdata["_region_idx"] = region_idxs

    def mesh(self):
        """Return Mesh object from xplt data"""
        # Get list of nodes as spatial coordinates.  According to the
        # FEBio binary database spec, there is only one `node coords`
        # section.
        node_data = get_bdata_by_name(
            self.blocks, f"{MESH_PATH[self.version]}/nodes/node coords"
        )[0]
        if self.version < 48:
            # XPLT 1.0
            x_nodes = [node_data[3 * i : 3 * i + 3] for i in range(len(node_data) // 3)]
        else:
            # XPLT 2.0
            n = get_bdata_by_name(
                self.blocks, f"{MESH_PATH[self.version]}/nodes/node header/size"
            )[0][0]
            values = struct.unpack(self.endian + n * "Ifff", node_data)
            node_ids = [values[i] for i in range(0, 4 * n, 4)]
            assert np.all(np.diff(node_ids) == 1)
            x_nodes = [values[i + 1 : i + 4] for i in range(0, 4 * n, 4)]
        # Get list of elements for each domain.
        b_domains = get_bdata_by_name(self.blocks, f"{MESH_PATH[self.version]}/domains")
        elements = []
        for b in b_domains:
            # Get list of elements as tuples of node ids.  Note that the
            # binary database format uses 0-indexing for nodes, same as
            # waffleiron.  The data field for each element's block
            # contains the element ID followed by the element's node
            # IDs.
            i_elements = get_bdata_by_name(b["data"], "element_list/element")
            element_ids = [r[0] for r in i_elements]
            i_elements = [r[1:] for r in i_elements]
            # Get material.  Note that the febio binary database
            # uses 1-indexing for element IDs.
            i_mat = find_one(b["data"], "domain_header/part ID")["data"] - 1
            # Get element type
            ecode = find_one(b["data"], "domain_header/element_type")["data"]
            etype = element_type_from_id[ecode]
            # Create list of element objects.  TODO: storing the ordinal
            # ID as the material is non-standard; figure out a better
            # way to handle the material information available in the
            # xplt.  Such as returning a list of domains.
            elements += [
                etype.from_ids(i_element, x_nodes, mat=i_mat)
                for i_element in i_elements
            ]
        mesh = Mesh(x_nodes, elements)
        return mesh

    def region_with_entity(self, entity_type, entity_id):
        """Return regions containing given entity.

        Returns a sequence of (region_type, region_id) tuples containing
        the given entity ID.

        """
        raise NotImplementedError

    def _entity_idx_in_sdata(
        self,
        var,
        entity_type,
        entity_id,
        val_type,
        layout="item",
        region_type=None,
        region_id=None,
        parent_id=None,
    ):
        """Return the index of an entity's value in step data.

        Returns an index equal to the offset in bytes from the start of
        the step data (including the first region ID word) to the value
        corresponding to the specified entity.

        """
        # Where an entity's value lies within step data depends on the
        # the entity itself, the region to use (entities may belong to
        # more than one region), the value's data format, the value's
        # layout, and the variable being accessed (values may be stored
        # for some regions but not others).
        if entity_type in ("domain", "surface", "region"):
            region_id = entity_id
        elif region_type == "node" and region_id is None:
            region_id = 0
        region_idx = self.variables[var]["_region_idx"][region_id]
        i = self._entity_idx_in_rdata(
            entity_type, entity_id, val_type, layout, region_type, region_id, parent_id
        )
        return region_idx + i

    def _entity_idx_in_rdata(
        self,
        entity_type,
        entity_id,
        val_type,
        layout,
        region_type,
        region_id=0,
        parent_id=None,
    ):
        """Return the index of an entity's value in regional data.

        entity_type is 'node', 'element', 'surface', or 'domain'.

        entity_id is a node ID, canonical face tuple, element ID, or
        region ID.  Node IDs and element IDs are 0-indexed.

        region_type is 'node', 'surface', or 'domain'.

        region_id is the ID of the region in which the entity should be
        looked up.  The default is 0, representing the whole mesh,
        albeit this is only valid for nodal data.  The region ID is
        required when the entity may belong to multiple regions.  In
        this case, the region ID disambiguates regarding which index is
        wanted.

        val_type is the value type ('int', 'float', or 'mat3fs').

        layout is how values are ordered within the step data ('node',
        'item', 'mult', or 'region').

        Returns an index equal to the offset in bytes from the start of
        the region data to the value corresponding to the specified entity.

        """
        # TODO: replace the entity_type and region_type parameters with
        # proper classes for the various IDs.
        #
        if entity_type in ("domain", "surface", "region"):
            return 0
        # o_idx := index to value of interest, counted by values
        if region_type == "node":
            o_idx = entity_id  # 0-indexed
        else:
            o_idx = self._regional_idx[region_type][region_id][layout][entity_id]
        if layout == "mult":
            # Apply parent entity ID to disambiguate
            o_idx = next(a[1] for a in o_idx if a[0] == parent_id)
        # b_idx := index to value of interest in sdata, counted by bytes
        b_idx = o_idx * VALUE_SZ_B[val_type]
        return b_idx

    def value(self, var, step, entity_id, region_id=None, parent_id=None):
        """Return a single value for a variable & selected entity.

        `entity_id` is a node ID (0-indexed), a canonical face tuple
        (containing 0-indexed node IDs), or an element ID (0-indexed).

        `region_id` is a plotfile-specific surface or domain ID.  It is
        used verbatim, which means it is 1-indexed.

        `parent_id` is either a canonical face tuple (using 0-indexed node
        IDs) or a 0-indexed element ID.

        """
        var_mdata = self.variables[var]
        iterator = _iter_step_data(
            (self.step_blocks[step],), self.variables[var], self.endian
        )
        # Make sure required selectors are present
        if var_mdata["region selector"] and region_id is None:
            raise ValueError(
                f"The variable '{var}' requires a `region_id` parameter ({var_mdata['region type']} ID) to disambiguate between {var_mdata['region type']}s."
            )
        if var_mdata["parent selector"] is not None and parent_id is None:
            raise ValueError(
                f"The variable '{var}' requires a `parent_id` parameter ({var_mdata['parent selector']} ID) to disambiguate between {var_mdata['parent selector']}s."
            )
        # Handle node variable case, which works differently than the
        # regional variables
        if var_mdata["region type"] == "node":
            idx = entity_id
            region_data = [d for d in iterator]
            time, c_region_id, values = region_data[0]
            if not c_region_id == 0:
                raise ValueError(
                    f"A region ID of {c_region_id} was encounted for variable {var}.  The FEBio plotfile spec states that node variables have region ID = 0, indicating the implicit region of all nodes."
                    + _PARSE_ERROR_GENERIC
                )
            if len(region_data) > 1:
                raise ValueError(
                    f"Node data with multiple regions was encountered for variable {var}.  Multiple regions are not expected for node data."
                    + _PARSE_ERROR_GENERIC
                )
            return values[idx]
        # Handle region layout case: no selection other than region ID
        if var_mdata["layout"] == "region":
            return next(d[2][0] for d in iterator if d[1] == entity_id)
        # Handle remaining regional (surface or domain) variable cases
        regional_idx = self._regional_idx[var_mdata["region type"]][region_id][
            var_mdata["layout"]
        ]
        try:
            idx = regional_idx[entity_id]
        except KeyError as err:
            raise type(err)(
                f"{var_mdata['entity type']} {entity_id} was not found in {var_mdata['region type']} {region_id}."
                + _LOOKUP_ERROR_GENERIC
            ) from err
        if var_mdata["layout"] == "node":
            idx = regional_idx[entity_id]
            return next(
                d[2][idx] for d in iterator if d[1] == region_id
            )  # TODO: the rdata should
            # be directly accessible
            # by region selector
        elif var_mdata["layout"] == "item":
            idx = regional_idx[entity_id]
            return next(d[2][idx] for d in iterator if d[1] == region_id)
        elif var_mdata["layout"] == "mult":
            try:
                idx = next(d[1] for d in idx if d[0] == parent_id)
            except StopIteration as err:
                raise ValueError(
                    f"No {var} value for {var_mdata['entity type']} {entity_id} was found for {var_mdata['parent selector']} {parent_id} in {var_mdata['region type']} {region_id}."
                    + _LOOKUP_ERROR_GENERIC
                ) from err
            return next(d[2][idx] for d in iterator if d[1] == region_id)
        else:
            raise ValueError(
                f"Variable {var} has a data layout type of '{var_mdata['layout']}', which is not recognized."
                + _PARSE_ERROR_GENERIC
            )

    def values(
        self, var, entity_id, region_id=None, parent_id=None, step_bounds=(0, inf)
    ):
        """Return time series values for a variable & selected entity."""
        # TODO: The representation of the time series should not require the user to
        #  type the variable name twice to get the variable's values.  This is not
        #  good: self.values("displacement", id_)["displacement"]
        mdata = self.variables[var]  # variable metadata dictionary
        if mdata["region type"] == "object":
            # FEBio treats object data differently from node, domain, or surface
            # data. There is a separate data block for each variable for each object.
            # However, get_state_var_data will concatenate variable blocks across
            # objects, so we still need a byte offset.
            offset = entity_id * VALUE_SZ_B[mdata["type"]]
        else:
            # Node, domain, or surface data
            # TODO: Check for sufficient selectors for this variable
            if mdata["parent selector"] is not None and parent_id is None:
                raise ValueError
            # Get indices to where the entity's value lies in the step data
            offset = self._entity_idx_in_sdata(
                var,
                mdata["entity type"],
                entity_id,
                mdata["type"],
                mdata["layout"],
                mdata["region type"],
                region_id,
                parent_id,
            )

        def unpacker(raw, mdata, endian=self.endian, offset=offset):
            return unpack_data(
                raw[offset : offset + VALUE_SZ_B[mdata["type"]]],
                mdata["type"],
                endian,
            )[0]

        # Iterate through steps, building up a time series table.
        columns = {"step": [], "time": [], var: []}
        step_idxs = range(
            max((0, step_bounds[0])), min((len(self.step_blocks), step_bounds[1]))
        )
        for step_idx in step_idxs:
            step_block = self.step_blocks[step_idx]
            raw = get_state_var_data(step_block, mdata)
            value = unpacker(raw, mdata)
            step_time = find_one(
                self.step_blocks[step_idx]["data"], "state header/time"
            )["data"]
            columns["step"].append(step_idx)
            columns["time"].append(step_time)
            columns[var].append(value)
        columns[var] = np.array(columns[var])
        return columns

    def step_data(self, idx):
        """Return all state data for a step

        The state data is returned as a map (entity type, variable name) → values.

        In FEBio terms, there are three types of state data in a step's
        state section:

        1. "State data" (the name collision is unfortunate) which covers
           data with four entity types: global, node, domain, and
           surface.

        2. Mesh data, which is new in FEBio 3.

        3. Object data, which is new in FEBio 3 and as of FEBio 3.5
           started to be used for rigid body data.

        Waffleiron does not yet support mesh data.

        Waffleiron treats object data merely as a another entity type,
        "object".

        Values are returned as a list of floats, vectors, or tensors
        according to the data type specified in the file's dictionary
        section.  The list index corresponds to the zero-indexed ID of
        the corresponding geometric entity.

        """
        data = {}
        state_block = self.step_blocks[idx]
        for varname, mdata in self.variables.items():
            raw = get_state_var_data(state_block, mdata)
            entity_type = mdata["region type"]
            if entity_type == "object":
                values = unpack_data(raw, val_type=mdata["type"], endian=self.endian)
            else:
                raw = get_state_var_data(state_block, mdata)
                # FEBio breaks from its documented tag format for
                # variable_data tags.  The data payload consists of, for
                # each region, a region ID (int; 4 bytes), the size of
                # the variable_data payload for that region enumerated
                # in bytes (int; 4 bytes), and the variable_data payload
                # for that region.
                i = 0  # offset into var['data'], in bytes
                values = []
                while i < len(raw):
                    # Iterate by region over variable_data data.
                    #
                    # TODO: What happens if regions don't have
                    # contiguous element numbering?  Do we need to
                    # correlate variable_data for element variables
                    # against element sets?  I don't think so; only
                    # nodesets are defined in the FEBio XML, so any new
                    # element ordering would have to be created de novo.
                    region_id, n = struct.unpack(self.endian + "II", raw[i : i + 8])
                    i += 8
                    values += unpack_data(
                        raw[i : i + n],
                        mdata["type"],
                        self.endian,
                    )
                    i += n
            data[(varname, entity_type)] = values
        # Add step time as a convenience
        data[("time", "global")] = self.step_times[idx]
        return data
