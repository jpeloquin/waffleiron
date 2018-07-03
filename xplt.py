# Base packages
import struct
import sys

# Same-package modules
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
    16920576: {'name': 'nodeset variables',  # 0x01023000
               'leaf': False},
    16924672: {'name': 'domain variables',  # 0x01024000
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
               'leaf': False},
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
    33686528: {'name': 'domain data',  # 0x02020400
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

item_format_from_id = {
    0: 'node',
    1: 'item',
    2: 'mult'
}


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


def parse_blocks(data, offset=0, store_data=True, endian='<'):
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
    i = 0
    blocks = []
    # Traverse the data, looking for blocks
    while i < len(data):
        # Get block id
        b_id = data[i:i+4]
        # Get block size
        b_size = data[i+4:i+8]
        i_size = struct.unpack(endian + 'I', b_size)[0]
        # Get this block's data/children
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
        if store_data:
            block['data'] = []

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
            block['data'] = parse_blocks(child, offset + i + 8,
                                         endian=endian, store_data=store_data)
        # Record this block's metadata and move on
        blocks.append(block)
        i += 8 + i_size
    return blocks


def parse_xplt(data, store_data=True):
    """Parse data as an FEBio binary database file.

    The parser is robust to unknown tags, but is not currently robust to
    misaligned tags or incomplete tags.

    """
    # Parse FEBio tag
    endian = parse_endianness(data[:4])

    # Parse the rest of the file
    parse_tree = parse_blocks(data[4:], offset=4, endian=endian, store_data=store_data)

    return parse_tree


def pprint_blocks(blocks, indent=0):
    s_indent = " " * (indent + 2)  # For second and following lines
    for i, block in enumerate(blocks):
        if i == 0:  # First block
            # Write opening braces
            sys.stdout.write(" " * indent + "[{")
        else:
            # Write the new block's indent & brace
            sys.stdout.write(" " * (indent + 1) + "{")
        sys.stdout.write("'name': '{}'".format(block['name']))
        for k in ['tag', 'type', 'address', 'size']:
            sys.stdout.write(",\n" + s_indent + "'{}': '{}'".format(k, block[k]))

        # Write block data/children
        if 'data' in block:
            if block['type'] == 'branch':
                sys.stdout.write(",\n" + s_indent + "'children':\n")
                pprint_blocks(block['data'], indent=indent+2)
            else:  # Leaf or unknown
                sys.stdout.write(",\n" + s_indent + "'data': {}".format(block['data']))

        if i < len(blocks) - 1:  # Before last block
            # Close the braces for the current block and write a
            # continuation comma
            print("},")
        else:  # Last block
            # Close the braces for the current block
            print("}")
            # Close the braces for the list of blocks
            sys.stdout.write(" " * indent + "]")


def unpack_block_data(data, fmt, endian):
    if fmt in ['int', 'float']:  # Numeric cases
        n = int(len(data) / 4)
        if fmt == 'int':
            v = struct.unpack(endian + n * 'I', data)
        else:
            v = struct.unpack(endian + n * 'f', data)
        if len(v) == 1:  # Flatten singleton tuple
            v = v[0]
        return v
    else:  # 'str' or 'variable' → pass through
        return data
