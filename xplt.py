# Base packages
from math import inf
import struct
import sys
from warnings import warn

# Third-party public packages
import numpy as np

# Same-package modules
from . import element, Mesh

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
    4605250: {'name': 'FEBio'},  # 0x00464542; not leaf or branch
    16777216: {'name': 'root',  # 0x01000000
               'leaf': False},
    33554432: {'name': 'state',  # 0x02000000
               'leaf': False},
    16842752: {'name': 'header',  # 0x01010000
               'leaf': False},
    16908288: {'name': 'dictionary',  # 0x01020000
               'leaf': False},
    16973824: {'name': 'materials',  # 0x01030000
               'leaf': False},
    17039360: {'name': 'geometry',  # 0x01040000
               'leaf': False},
    # Header section tags
    16842753: {'name': 'version',  # 0x01010001
               'leaf': True,
               'format': 'int'},
    16842754: {'name': 'nodes',  # 0x01010002
               'leaf': True,
               'format': 'int'},
    16842755: {'name': 'max facet nodes',  # 0x01010003
               'leaf': True,
               'format': 'int'},
    16842756: {'name': 'compression',  # 16842756
               'leaf': True},
    # Dictionary section tags; all optional
    16912384: {'name': 'global variables',  # 0x01021000
               'leaf': False},
    16916480: {'name': 'material variables',  # 0x01022000
               'leaf': False},
    16920576: {'name': 'node variables',  # 0x01023000
               'leaf': False},
    16924672: {'name': 'element variables',  # 0x01024000
               'leaf': False},
    16928768: {'name': 'surface variables',  # 0x01025000
               'leaf': False},
    16908289: {'name': 'dictionary item',  # 0x01020001
               'leaf': False},
    # Dictionary item tags
    16908290: {'name': 'item type',  # 0x01020002,
               'leaf': True,
               'format': 'int'},
    16908291: {'name': 'item format',  # 0x01020003
               'leaf': True,
               'format': 'int'},
    16908292: {'name': 'item name',  # 0x01020004
               'leaf': True,
               'format': 'str'},
    0x01020005: {'name': 'item array size',
                 'leaf': True},
    0x01020006: {'name': 'item_array name',
                 'leaf': True},
    # Root/Materials section tags
    16973825: {'name': 'material',  # 0x01030001
               'leaf': False},
    16973826: {'name': 'material id',  # 0x01030002
               'leaf': True,
               'format': 'int'},
    16973827: {'name': 'material name',  # 0x01030003
               'leaf': True,
               'format': 'str'},
    # Root/Geometry tags
    17043456: {'name': 'nodes',  # 0x01041000
               'leaf': False},
    17047552: {'name': 'domains',  # 0x01042000
               'leaf': False},
    17051648: {'name': 'surfaces',  # 0x01043000
               'leaf': False},
    17055744: {'name': 'nodesets',  # 0x01044000
               'leaf': False},
    # Root/Geometry/Node tags
    17043457: {'name': 'node coords',  # 0x01041001
               'leaf': True,
               'format': 'float'},
    # Root/Geometry/Domains tags
    17047808: {'name': 'domain',  # 0x01042100
               'leaf': False},
    # Root/Geometry/Domains/Domain tags
    17047809: {'name': 'domain_header',  # 0x01042101
               'leaf': False},
    17048064: {'name': 'element_list',  # 0x01042200
               'leaf': False},
    # Root/Geometry/Domains/Domain/Domain Header tags
    17047810: {'name': 'elem_type',  # 0x01042102
               'leaf': True,
               'format': 'int'},
    17047811: {'name': 'material ID',  # 0x01042103
               'leaf': True,
               'format': 'int'},
    16982276: {'name': 'elements',  # 0x01032104
               'leaf': True,
               'format': 'int'},
    16982277: {'name': 'domain name',  # 0x01032105
               'leaf': True,
               'format': 'str'},
    # Root/Geometry/Domains/Domain/Element List tags
    17048065: {'name': 'element',  # 0x01042201
               'leaf': True,
               'format': 'int'},
    # Root/Geometry/Surfaces tags
    17051904: {'name': 'surface',  # 0x01043100
               'leaf': False},
    # Root/Geometry/Surfaces/Surface tags
    17051905: {'name': 'surface header',  # 0x01043101
               'leaf': False},
    17052160: {'name': 'facet list',  # 0x01043200
               'leaf': False},
    # Root/Geometry/Surfaces/Surface/Surface Header tags
    17051906: {'name': 'surface_id',  # 0x01043102
               'leaf': True,
               'format': 'int'},
    17051907: {'name': 'facets',  # 0x01043103
               'leaf': True,
               'format': 'int'},
    17051908: {'name': 'surface name',  # 0x01043104
               'leaf': True,
               'format': 'str'},
    # Root/Geometry/Surfaces/Surface/Facet List tags
    17052161: {'name': 'facet',  # 0x01043201
               'leaf': True,
               'format': 'int'},
    # Root/Geometry/Nodesets tags
    17056000: {'name': 'nodeset',  # 0x01044100
               'leaf': False},
    # Root/Geometry/Nodesets/Nodeset tags
    17056001: {'name': 'nodeset header',  # 0x01044101
               'leaf': False},
    17056256: {'name': 'node list',  # 0x01044200
               'leaf': True,
               'format': 'int'},
    # Root/Geometry/Nodesets/Nodeset Header tags
    17056002: {'name': 'nodeset ID',  # 0x01044102
               'leaf': True,
               'format': 'int'},
    17056004: {'name': 'nodes',  # 0x01044104
               'leaf': True,
               'format': 'int'},
    17056003: {'name': 'nodeset name',  # 0x01044103
               'leaf': True,
               'format': 'str'},
    # State tags
    33619968: {'name': 'state header',  # 0x02010000
               'leaf': False},
    33685504: {'name': 'state data',  # 0x02020000
               'leaf': False},
    # State/State Header tags
    33619970: {'name': 'time',  # 0x02010002
               'leaf': True,
               'format': 'float'},
    # State/State Data tags
    33685760: {'name': 'global data',  # 0x02020100
               'leaf': False},
    33686016: {'name': 'material data',  # 0x02020200
               'leaf': False},
    33686272: {'name': 'node data',  # 0x02020300
               'leaf': False},
    33686528: {'name': 'element data',  # 0x02020400
               'leaf': False},
    33686784: {'name': 'surface data',  # 0x02020500
               'leaf': False},
    # State/State Data/* tags
    33685505: {'name': 'state variable',  # 0x02020001
               'leaf': False},
    # State/State Data/*/State Variable tags
    33685506: {'name': 'variable ID',  # 0x02020002
               'leaf': True,
               'format': 'int'},
    33685507: {'name': 'data',  # 0x02020003
               'leaf': True,
               'format': 'variable'}
}

element_type_from_id = {
    0: element.Hex8,
    1: element.Penta6,
    2: 'tet4',
    3: element.Quad4,
    4: element.Tri3,
    5: 'truss2'
}

item_type_from_id = {
    0: 'float',
    1: 'vec3f',
    2: 'mat3fs'
}

item_format_from_id = {  # Refer to FE_enum.h:326
    0: 'node',
    1: 'item',
    2: 'mult',
    3: 'region'
}


dict_section_with_var = {
    "acceleration": "element variables",
    "contact area": "surface variables",
    "contact force": "surface variables",
    "contact gap": "surface variables",
    "contact penalty": "surface variables",
    "contact pressure": "surface variables",
    "contact stick": "surface variables",
    "contact traction": "surface variables",
    "current density": "element variables",
    "current element angular momentum": "element variables",
    "current element center of mass": "element variables",
    "current element kinetic energy": "element variables",
    "current element linear momentum": "element variables",
    "current element strain energy": "element variables",
    "damage": "element variables",
    "density": "element variables",
    "deviatoric strain energy density": "element variables",
    "displacement": "node variables",
    "effective elasticity": "element variables",
    "effective fluid pressure": "node variables",
    "effective shell fluid pressure": "node variables",
    "effective shell solute concentration": "node variables",
    "effective solute concentration": "node variables",
    "effective fluid pressure": "node variables",
    "effective solute concentration": "node variables",
    "elasticity": "element variables",
    "electric potential": "element variables",
    "element angular momentum": "element variables",
    "element center of mass": "element variables",
    "element kinetic energy": "element variables",
    "element linear momentum": "element variables",
    "element strain energy": "element variables",
    "element stress power": "element variables",
    "enclosed volume": "surface variables",
    "Euler angle": "element variables / rigid body",
    "fiber stretch": "element variables",
    "fiber vector": "element variables",
    "fixed charge density": "element variables",
    "fluid acceleration": "element variables",
    "fluid density": "element variables",
    "fluid dilation": "node variables",
    "fluid element angular momentum": "element variables",
    "fluid element center of mass": "element variables",
    "fluid element kinetic energy": "element variables",
    "fluid element linear momentum": "element variables",
    "fluid element strain energy": "element variables",
    "fluid energy density": "element variables",
    "fluid flow rate": "surface variables",
    "fluid flux": "element variables",
    "fluid force": "surface variables",
    "fluid force2": "surface variables",
    "fluid heat supply density": "element variables",
    "fluid kinetic energy density": "element variables",
    "fluid load support": "surface variables",
    "fluid mass flow rate": "surface variables",
    "fluid pressure": "element variables",
    "fluid rate of deformation": "element variables",
    "fluid shear viscosity": "element variables",
    "fluid strain energy density": "element variables",
    "fluid stress": "element variables",
    "fluid stress power density": "element variables",
    "fluid surface energy flux": "surface variables",
    "fluid surface force": "surface variables",
    "fluid surface traction power": "surface variables",
    "fluid velocity": "element variables",
    "fluid volume ratio": "element variables",
    "fluid vorticity": "element variables",
    "heat flux": "element variables",
    "kinetic energy density": "element variables",
    "Lagrange strain": "element variables",
    "nested damage": "element variables",
    "nodal acceleration": "node variables",
    "nodal contact gap": "surface variables",
    "nodal contact pressure": "surface variables",
    "nodal contact traction": "surface variables",
    "nodal fluid flux": "element variables",
    "nodal fluid velocity": "node variables",
    "nodal stress": "element variables",  # nonstandard; has way too many values
    "nodal surface traction": "surface variables",
    "nodal vector gap": "surface variables",
    "nodal velocity": "node variables",
    "material axes": "element variables",  # undocumented
    "osmolarity": "element variables",
    "parameter": "element variables",
    "pressure gap": "surface variables",
    "reaction forces": "node variables",
    "receptor-ligand concentration": "element variables",
    "referential fixed charge density": "element variables",
    "referential solid volume fraction": "element variables",
    "relative fluid velocity": "element variables",
    "relative volume": "element variables",  # nonstandard; has way too many values
    "rigid acceleration": "element variables / rigid body",
    "rigid angular acceleration": "element variables / rigid body",
    "rigid angular momentum": "element variables / rigid body",
    "rigid angular position": "element variables / rigid body",
    "rigid angular velocity": "element variables / rigid body",
    "rigid force": "element variables / rigid body",
    "rigid kinetic energy": "element variables / rigid body",
    "rigid linear momentum": "element variables / rigid body",
    "rigid position": "element variables / rigid body",
    "rigid rotation vector": "element variables / rigid body",
    "rigid torque": "element variables / rigid body",
    "rigid velocity": "element variables / rigid body",
    "sbm concentration": "element variables",
    "sbm referential apparent density": "element variables",
    "shell director": "element variables",
    "shell relative volume": "element variables",
    "shell strain": "element variables",
    "shell thickness": "node variables",
    "solute concentration": "element variables",
    "solute flux": "element variables",
    "specific strain energy": "element variables",
    "SPR Lagrange strain": "element variables",  # nonstandard; has way too many values
    "SPR principal stress": "element variables",
    "SPR stress": "element variables",
    "SPR-P1 stress": "element variables",
    "strain energy density": "element variables",
    "stress": "element variables",
    "surface traction": "surface variables",
    "uncoupled pressure": "element variables",
    "ut4 nodal stress": "element variables",
    "vector gap": "element variables",
    "velocity": "node variables",
    "volume fraction": "element variables"}


def parse_endianness(data):
    """Return endianness of xplt data.

    """
    # Parse identifier tag
    if data[:4] == b'BEF\x00':
        endian = '<'
    elif data[:4] == b'\x00FEB':
        endian = '>'
    else:
        msg = "Input data is not valid as an FEBio binary database file.  "\
            "The first four bytes are {}, but should be 'BEF\x00' or "\
            "'\x00F\EB'."
        raise ValueError(msg.format(data[:4]))

    return endian


def parse_blocks(data, offset=0, store_data=True, max_depth=inf, endian='<'):
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
            warn("A branch node's block contains data that does not unpack into a valid child node.  This data will be skipped.")
            break
        # Get block id
        b_id = data[i:i+4]
        # Get block size
        b_size = data[i+4:i+8]
        i_size = struct.unpack(endian + 'I', b_size)[0]
        # Get this block's data/children
        if i + i_size > len(data):
            warn("A branch node's block contains data that does not unpack into a valid child node.  This data will be skipped.")
            break
        child = data[i+8:i+8+i_size]
        # Convert ID from bytes to nicer form
        i_id = struct.unpack(endian + 'I', b_id)[0]
        s_id = "0x" + format(i_id, '08x')
        # Collect the block's basic metadata.
        block = {'name': 'unknown',
                 'tag': s_id,
                 'type': 'unknown',
                 'address': i + offset,
                 'size': i_size}

        # Try looking up specification metadata.  If we fail, treat this
        # block as a leaf and return the basic metadata.
        try:
            tag_metadata = tags_table[i_id]
        except KeyError:
            if store_data:
                block['data'] = child
            blocks.append(block)
            i += 8 + i_size
            continue
        block['name'] = tag_metadata['name']
        # If this is a leaf block, parse its data.  If it is a branch
        # block, parse its children.
        if tag_metadata['leaf']:  # is a leaf block
            block['type'] = 'leaf'
            if 'format' in tag_metadata:
                fmt = tag_metadata['format']
                if store_data:
                    block['data'] = unpack_block_data(child, fmt, endian)
            else:
                if store_data:
                    block['data'] = child
        else:  # is a branch block
            block['type'] = 'branch'
            if max_depth > 1:
                block['data'] = parse_blocks(child, offset=offset + i + 8, endian=endian,
                                             store_data=store_data,
                                             max_depth=max_depth - 1)
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


def pprint_blocks(f, blocks, indent=0):
    """Pretty-print parsed XPLT blocks.

    `f` must support a `write` method.

    """
    s_indent = " " * (indent + 2)  # For second and following lines
    for i, block in enumerate(blocks):
        if i == 0:  # First block
            # Write opening braces
            f.write(" " * indent + "[{")
        else:
            # Write the new block's indent & brace
            f.write(" " * (indent + 1) + "{")
        f.write("'name': '{}'".format(block['name']))
        for k in ['tag', 'type', 'address', 'size']:
            f.write(",\n" + s_indent + "'{}': '{}'".format(k, block[k]))

        # Write block data/children
        if 'data' in block:
            if block['type'] == 'branch':
                f.write(",\n" + s_indent + "'children':\n")
                pprint_blocks(f, block['data'], indent=indent+2)
            else:  # Leaf or unknown
                f.write(",\n" + s_indent + "'data': {}".format(block['data']))

        if i < len(blocks) - 1:  # Before last block
            # Close the braces for the current block and write a
            # continuation comma
            f.write("},\n")
        else:  # Last block
            # Close the braces for the current block
            f.write("}\n")
            # Close the braces for the list of blocks
            f.write(" " * indent + "]")


def unpack_block_data(data, fmt, endian):
    if fmt in ['int', 'float']:  # Numeric cases
        n = int(len(data) / 4)
        if fmt == 'int':
            v = struct.unpack(endian + n * 'I', data)
        else:  # float
            v = struct.unpack(endian + n * 'f', data)
    elif fmt == 'vec3f':
        if len(data) % 12 != 0:
            raise ValueError("Input data cannot be evenly divided into "
                             "vectors.")
        v = []
        for j in range(0, len(data), 12):
            v.append(np.array(struct.unpack(endian + 'f' * 3,
                                            data[j:j+12])))
    elif fmt == 'mat3fs':
        if len(data) % 24 != 0:
            raise ValueError("Input data cannot be evenly divided into "
                             "tensor.")
        v = []
        for j in range(0, len(data), 24):
            a = struct.unpack(endian + 'f' * 6, data[j:j+24])
            # The FEBio database spec does not document the
            # tensor order, but this should be accurate.
            v.append(np.array([[a[0], a[3], a[5]],
                               [a[3], a[1], a[4]],
                               [a[5], a[4], a[2]]]))
    elif fmt == 'str':
        # FEBio uses null-terminated strings.  Strip the null byte and
        # everything after it, and decode from bytes to text.
        v = data[0:data.find(b'\x00')].decode()
    else:
        # Pass through unrecognized values.
        v = data
    return v


class XpltBlocks:
    """Xplt parse structure with accessor methods.

    """
    def __init__(self, blocks):
        self.blocks = blocks

    def __repr__(self):
        return repr(self.blocks)

    def get_all(self, pth):

        for key in keys:
            matches = [b['data'] for b in self.blocks if b['name'] == key]
        return XpltBlocks(matches)


def get_bdata_by_name(blocks, pth):
    """Return a list of the contents of all blocks matching a path.

    """
    if type(blocks) is not  list:
        blocks = [blocks]
    names = pth.split('/')
    while len(names) != 0:
        name, names = names[0], names[1:]
        matches = []
        for b in blocks:
            if b['name'] == name:
                if b['type'] == 'branch':
                    matches += b['data']
                elif b['type'] == 'leaf' and len(names) == 0:
                    # Leaf blocks can only match the end of the given
                    # name path.
                    matches.append(b['data'])
        blocks = matches
    return matches


class XpltData:
    """In-memory storage and reading of xplt file data.

    """
    def __init__(self, data):
        """Initialize XpltData object from xplt bytes data.

        """
        self.endian = parse_endianness(data[:4])
        blocks = parse_xplt_data(data, store_data=True)
        # Store header data
        self.header_blocks = blocks[0]['data']
        # Store step data
        self.step_blocks = blocks[1:]
        # Step times
        self.step_times = []
        for b in self.step_blocks:
            t = get_bdata_by_name(b['data'], 'state header/time')[0]
            self.step_times.append(t)
        # Step data dictionary
        b_data_dictionary = get_bdata_by_name(self.header_blocks, 'dictionary')
        self.data_dictionary = {}
        for b_cat in b_data_dictionary:
            for b_var in b_cat['data']:
                var = {}
                for b in b_var['data']:
                    # b is a block with keys: "name", "tag", "type" →
                    # "leaf" or "branch", "address" → int, "size" → int,
                    # and "data"
                    var[b['name']] = b['data']
                # Flatten data that is supposed to have only one value
                assert len(var["item type"]) == 1
                var['item type'] = var['item type'][0]
                assert len(var["item format"]) == 1
                var['item format'] = var['item format'][0]
                # Convert coded values
                var['item type'] = item_type_from_id[var['item type']]
                var['item format'] = item_format_from_id[var['item format']]
                # Append variable entry to its category in the main data
                # dictionary.
                self.data_dictionary.setdefault(b_cat['name'], []).append(var)

    def material_names(self):
        """Return dict of material IDs → names.

        """
        pass

    def mesh(self):
        # Get list of nodes as spatial coordinates.  According to the
        # FEBio binary database spec, there is only one `node coords`
        # section.
        node_data = get_bdata_by_name(self.header_blocks,
            'geometry/nodes/node coords')[0]
        x_nodes = [node_data[3*i:3*i+3] for i in range(len(node_data) // 3)]
        # Get list of elements for each domain.
        b_domains = get_bdata_by_name(self.header_blocks, 'geometry/domains')
        elements = []
        for b in b_domains:
            # Get list of elements as tuples of node ids.  Note that the
            # binary database format uses 0-indexing for nodes, same as
            # febtools.  The data field for each element's block
            # contains the element ID followed by the element's node
            # IDs.
            i_elements = get_bdata_by_name(b['data'], 'element_list/element')
            element_ids = [r[0] for r in i_elements]
            i_elements = [r[1:] for r in i_elements]
            # Get material.  Note that the febio binary database
            # uses 1-indexing for element IDs.
            i_mat = get_bdata_by_name(b['data'], 'domain_header/material ID')[0][0] - 1
            # Get element type
            ecode = get_bdata_by_name(b['data'], 'domain_header/elem_type')[0][0]
            etype = element_type_from_id[ecode]
            # Create list of element objects
            elements += [etype.from_ids(i_element, x_nodes, mat_id=i_mat)
                         for i_element in i_elements]
        mesh = Mesh(x_nodes, elements)
        return mesh

    def step_data(self, idx):
        """Retrieve data for a specific solution step.

        The solution data is returned as a dictionary.  The data names
        (e.g. stress, displacement) are the keys.  These keys are read
        from the file's dictionary section.  Data is formatted into a
        list of floats, vectors, or tensors according to the data type
        specified in the file's dictionary section.

        """
        data = {}
        b_statedata = get_bdata_by_name(self.step_blocks[idx]['data'], 'state data')
        for b_cat in b_statedata:  # iterate over data category blocks
            base_name = b_cat['name'].split(' ')[0]
            cat_name = base_name + ' variables'
            for b_var in b_cat['data']:  # iterate over state variable blocks
                var = {}
                for b in b_var['data']:
                    var[b['name']] = b['data']
                # Unpack variable data
                assert len(var["variable ID"]) == 1
                var["variable ID"] = var["variable ID"][0]
                var_id = var['variable ID'] - 1  # to 0-index
                entry = self.data_dictionary[cat_name][var_id]
                # FEBio breaks from its documented tag format for
                # variable_data tags.  The data payload consists of, for
                # each region, a region ID (int; 4 bytes), the size of
                # the varible_data payload for that region enumerated in
                # bytes (int; 4 bytes), and the variable_data payload
                # for that region.
                i = 0  # offset into var['data'], in bytes
                values = []
                while i < len(var['data']):
                    # Iterate by region over variable_data data.
                    #
                    # TODO: What happens if regions don't have
                    # contiguous element numbering?  Do we need to
                    # correlate variable_data for element variables
                    # against element sets?  I don't think so; only
                    # nodesets are defined in the FEBio XML, so any new
                    # element ordering would have to be created de novo.
                    region_id, n = struct.unpack(self.endian + 'II',
                                                 var['data'][i:i+8])
                    i += 8
                    values += unpack_block_data(var['data'][i:i+n],
                                                fmt=entry['item type'],
                                                endian=self.endian)
                    i += n
                data.setdefault(cat_name, {})[entry['item name']] = values

        # Add step time as a convenience
        data['time'] = self.step_times[idx]

        return data
