# Coordinates system: Left-handed
# Axes setup: X (right), Y (up), Z (front)
# Units: Meters

bl_info = {
    "name": "Cas2 Exporter",
    "author": "Mr.Jox (victimized.)",
    "version": (1, 0, 0),
    "blender": (4, 3, 1),
    "category": "Export",
}

import bpy
import struct
import os
import bmesh
import numpy as np
from datetime import datetime
from enum import Enum
from bpy import context
from mathutils import Vector

global_scene = context.scene

CONST_INT8_MIN = -0x80          # -128
CONST_INT8_MAX = 0x7F           # 127
CONST_UINT8_MIN = 0             # 0
CONST_UINT8_MAX = 0xFF          # 255

CONST_INT16_MIN = -0x8000       # -32768
CONST_INT16_MAX = 0x7FFF        # 32767
CONST_UINT16_MIN = 0            # 0
CONST_UINT16_MAX = 0xFFFF       # 65535

CONST_INT32_MIN = -0x80000000   # -2,147,483,648
CONST_INT32_MAX = 0x7FFFFFFF    # 2,147,483,647
CONST_UINT32_MIN = 0            # 0
CONST_UINT32_MAX = 0xFFFFFFFF   # 4,294,967,295

CONST_FLOAT_MAX = np.float32(2.54e+28)
CONST_FLOAT_MIN = np.float32(-2.54e+28)

CONST_UINT16_SIZE = struct.calcsize('H')
CONST_INT16_SIZE = struct.calcsize('h')
CONST_UINT16_SIZE = struct.calcsize('H')
CONST_UINT32_SIZE = struct.calcsize('I')
CONST_INT32_SIZE = struct.calcsize('i')
CONST_FLOAT_SIZE = struct.calcsize('f')

CONST_OBJECT_PROPERTY_PREFIX = "property_"
CONST_OBJECT_ATTRIBUTE_PREFIX = "attribute_"
CONST_ATTRIBUTES_STRING = "string_attributes"
CONST_ATTRIBUTES_INT = "int_attributes"
CONST_ATTRIBUTES_FLOAT = "float_attributes"
CONST_ATTRIBUTES_VEC3 = "vec3_attributes"
CONST_ATTRIBUTES_VEC4 = "vec4_attributes"

class NodeType(Enum):
    CAMERA = 6
    RIGID_MODEL = 7
    WEIGHTED_MODEL = 10
    LINE = 11
    SCENE_ROOT = 12
    MATERIAL = 13
    INSTANCE_NO_MATERIAL = 16
    DUMMY = 17
    INSTANCE_OVERRIDE_MATERIAL = 18

class AttributeType(Enum):
    FLOAT = 0
    STRING = 1
    VEC3 = 3
    INT32 = 8
    VEC4 = 9

class GeometryDataType(Enum):
    UNSET = -1
    MESH = 0
    LINE = 1

class Utf16String:
    def __init__(self, value = ""):
        self.string = value

    def write(self, file):
        utf16_string_length = len(self.string)
        utf16_string_bytes = self.string.encode('utf-16-le')
        utf16_string_data = pack_uint16(utf16_string_length) + utf16_string_bytes
        file.write(utf16_string_data)

    def read(self, file):
        length_data = file.read(2)
        utf16_string_length = struct.unpack('H', length_data)[0]
        utf16_string_bytes = file.read(utf16_string_length)
        self.string = utf16_string_bytes.decode('utf-16-le')
    
    def size(self):
        return CONST_UINT16_SIZE + len(self.string) * 2

class Vec2:
    def __init__(self, x = 0.0, y = 0.0):
        self.x = x
        self.y = y

    def write(self, file):
        file.write(pack_float(self.x))
        file.write(pack_float(self.y))

    def read(self, file):
        pass

    def size(self):
        return 2 * CONST_FLOAT_SIZE

class Vec3:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def write(self, file):
        file.write(pack_float(self.x))
        file.write(pack_float(self.y))
        file.write(pack_float(self.z))

    def read(self, file):
        pass

    def size(self):
        return 3 * CONST_FLOAT_SIZE

class Vec4:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0, w = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def write(self, file):
        file.write(pack_float(self.x))
        file.write(pack_float(self.y))
        file.write(pack_float(self.z))
        file.write(pack_float(self.w))

    def read(self, file):
        pass

    def size(self):
        return 4 * CONST_FLOAT_SIZE

class Quaternion:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0, w = 1.0):
        self.vec4 = Vec4(x, y, z, w)

    def write(self, file):
        self.vec4.write(file)

    def read(self, file):
        pass

    def size(self):
        return self.vec4.size()

class QuatVec3Transform:
    def __init__(self):
        self.rotation = Quaternion()
        self.translation = Vec3()

    def write(self, file):
        self.rotation.write(file)
        self.translation.write(file)

    def read(self, file):
        pass

    def size(self):
        return self.rotation.size() + self.translation.size()

class AttributeInfo:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class StringAttribute:
    def __init__(self, name, value):
        self.name = Utf16String(name)
        self.type = AttributeType.STRING
        self.string_value = Utf16String(value)

    def write(self, file):
        self.name.write(file)
        file.write(pack_uint32(self.type.value))
        self.string_value.write(file)

    def read(self, file):
        pass

    def size(self):
        return self.name.size() + CONST_UINT32_SIZE + self.string_value.size()

class IntAttribute:
    def __init__(self, name, value):
        self.name = Utf16String(name)
        self.unknown_1 = 0
        self.type = AttributeType.INT32
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_4 = 0
        self.unknown_5 = 0
        self.unknown_6 = 0
        self.int_value = value

    def write(self, file):
        self.name.write(file)
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.type.value))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.unknown_3))
        file.write(pack_uint32(self.unknown_4))
        file.write(pack_uint32(self.unknown_5))
        file.write(pack_uint32(self.unknown_6))
        file.write(pack_int32(self.int_value))

    def read(self, file):
        pass

    def size(self):
        return CONST_UINT32_SIZE * 7 + CONST_INT32_SIZE + self.name.size()

class FloatAttribute:
    def __init__(self, name, value):
        self.name = Utf16String(name)
        self.unknown_1 = 0
        self.type = AttributeType.FLOAT
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_4 = 0
        self.unknown_5 = 0
        self.unknown_6 = 0
        self.float_value = value

    def write(self, file):
        self.name.write(file)
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.type.value))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.unknown_3))
        file.write(pack_uint32(self.unknown_4))
        file.write(pack_uint32(self.unknown_5))
        file.write(pack_uint32(self.unknown_6))
        file.write(pack_float(self.float_value))

    def read(self, file):
        pass

    def size(self):
        return CONST_UINT32_SIZE * 7 + CONST_FLOAT_SIZE + self.name.size()

class Vec3Attribute:
    def __init__(self, name, value):
        self.name = Utf16String(name)
        self.unknown_1 = 0
        self.type = AttributeType.VEC3
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_4 = 0
        self.unknown_5 = 0
        self.unknown_6 = 0
        self.vec3_value = value

    def write(self, file):
        self.name.write(file)
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.type.value))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.unknown_3))
        file.write(pack_uint32(self.unknown_4))
        file.write(pack_uint32(self.unknown_5))
        file.write(pack_uint32(self.unknown_6))
        self.vec3_value.write(file)

    def read(self, file):
        pass

    def size(self):
        return CONST_UINT32_SIZE * 7 + self.vec3_value.size() + self.name.size()

class Vec4Attribute:
    def __init__(self, name, value):
        self.name = Utf16String(name)
        self.unknown_1 = 0
        self.type = AttributeType.VEC4
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_4 = 0
        self.unknown_5 = 0
        self.unknown_6 = 0
        self.vec4_value = value

    def write(self, file):
        self.name.write(file)
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.type.value))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.unknown_3))
        file.write(pack_uint32(self.unknown_4))
        file.write(pack_uint32(self.unknown_5))
        file.write(pack_uint32(self.unknown_6))
        self.vec4_value.write(file)

    def read(self, file):
        pass

    def size(self):
        return CONST_UINT32_SIZE * 7 + self.vec4_value.size() + self.name.size()

class NodeAttributes:
    def __init__(self, obj):
        attributes = build_attributes(obj)

        self.block_size = 0
        self.string_attributes = attributes[CONST_ATTRIBUTES_STRING]
        self.int_attributes = attributes[CONST_ATTRIBUTES_INT]
        self.float_attributes = attributes[CONST_ATTRIBUTES_FLOAT]
        self.vec3_attributes = attributes[CONST_ATTRIBUTES_VEC3]
        self.vec4_attributes = attributes[CONST_ATTRIBUTES_VEC4]

    def write_attributes_array(self, file, attributes):
        file.write(pack_uint32(len(attributes)))
        self.block_size += 4 # length size

        for attribute in attributes:
            self.block_size += attribute.size()
            attribute.write(file)

    def write(self, file):
        self.write_attributes_array(file, self.string_attributes)
        self.write_attributes_array(file, self.int_attributes)
        self.write_attributes_array(file, self.float_attributes)
        self.write_attributes_array(file, self.vec3_attributes)
        self.write_attributes_array(file, self.vec4_attributes)

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class UnknownKeyFramesBlockStruct1:
    def __init__(self):
        self.unknown_count = 0
        self.struct_size = 0
        self.unknown_array_1_size = 0
        self.unknown_array_2_size = 0
        self.unknown_array_1 = []
        self.unknown_array_2 = []

    def fill_temp_data(self):
        self.unknown_count = 1

        self.unknown_array_1_size = 2
        self.unknown_array_2_size = 1

        self.struct_size = 12

        for i in range(self.unknown_count):
            self.unknown_array_1 = [0 for _ in range(self.unknown_array_1_size)]
            self.unknown_array_2 = [0 for _ in range(self.unknown_array_2_size)]

        self.struct_size += self.unknown_array_1_size * CONST_UINT32_SIZE
        self.struct_size += self.unknown_array_2_size * CONST_UINT32_SIZE

    def write(self, file):
        file.write(pack_uint32(self.unknown_count))

        for i in range(self.unknown_count):
            file.write(pack_uint32(len(self.unknown_array_1)))
            for j in range(self.unknown_array_1_size):
                file.write(pack_uint32(self.unknown_array_1[j]))

            file.write(pack_uint32(len(self.unknown_array_2)))
            for j in range(self.unknown_array_2_size):
                file.write(pack_uint32(self.unknown_array_2[j]))

    def read(self, file):
        pass

    def size(self):
        return self.struct_size

class UnknownKeyFramesBlockStruct2:
    def __init__(self):
        self.unknown_count = 0
        self.struct_size = 0
        self.unknown_array = []

    def fill_temp_data(self):
        self.unknown_count = 1
        self.struct_size = 4
        self.unknown_array = [QuatVec3Transform() for _ in range(self.unknown_count)]

        for i in range(len(self.unknown_array)):
            self.struct_size += self.unknown_array[i].size()

    def write(self, file):
        file.write(pack_uint32(self.unknown_count))

        for i in range(len(self.unknown_array)):
            self.unknown_array[i].write(file)

    def read(self, file):
        pass

    def size(self):
        return self.struct_size

class UnknownKeyFramesBlockStruct3:
    def __init__(self):
        self.unknown_count = 0
        self.struct_size = 0
        self.unknown_1 = 0
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_array_1_count = 0
        self.unknown_array_2_count = 0
        self.unknown_array_1 = []
        self.unknown_array_2 = []

    def fill_temp_data(self):
        self.unknown_count = 0
        self.struct_size = 4 # unknown_count size

        self.unknown_array_1_count = 10
        self.unknown_array_2_count = 10

        for i in range(self.unknown_count):
            self.unknown_1 = 8
            self.unknown_2 = 0
            self.unknown_3 = 0
            self.struct_size += 20 # unknown_1 + unknown_2 + unknown_3 + unknown_array_1_count + unknown_array_2_count

            self.unknown_array_1 = [0.0 for _ in range(self.unknown_array_1_count)]
            self.struct_size += CONST_FLOAT_SIZE * len(self.unknown_array_1)

            self.unknown_array_2 = [-1 for _ in range(self.unknown_array_2_count)]
            self.struct_size += CONST_INT32_SIZE * len(self.unknown_array_2)

    def write(self, file):
        file.write(pack_uint32(self.unknown_count))

        for i in range(self.unknown_count):
            file.write(pack_uint32(self.unknown_1))
            file.write(pack_uint32(self.unknown_2))
            file.write(pack_uint32(self.unknown_3))

            file.write(pack_uint32(self.unknown_array_1_count))
            for j in range(self.unknown_array_1_count):
                file.write(pack_float(self.unknown_array_1[j]))

            file.write(pack_uint32(self.unknown_array_2_count))
            for j in range(self.unknown_array_2_count):
                file.write(pack_int32(self.unknown_array_2[j]))

    def read(self, file):
        pass

    def size(self):
        return self.struct_size

class UnknownSceneRootData1:
    def __init__(self):
        self.unknown_1 = 3
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_4 = 1
        self.unknown_5 = 0
        self.unknown_6 = 1
        self.unknown_7 = 0
        self.unknown_8 = 0
        self.unknown_9 = 0
        self.unknown_10 = 1
        self.unknown_11 = 6
        self.unknown_12 = 0
        self.unknown_13 = 0
        self.unknown_14 = 1
        self.unknown_15 = 0
        self.unknown_16 = 1
        self.unknown_quaternion = Quaternion()

    def write(self, file):
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.unknown_3))
        file.write(pack_uint32(self.unknown_4))
        file.write(pack_uint32(self.unknown_5))
        file.write(pack_uint32(self.unknown_6))
        file.write(pack_uint32(self.unknown_7))
        file.write(pack_uint32(self.unknown_8))
        file.write(pack_uint32(self.unknown_9))
        file.write(pack_uint32(self.unknown_10))
        file.write(pack_uint32(self.unknown_11))
        file.write(pack_uint32(self.unknown_12))
        file.write(pack_uint32(self.unknown_13))
        file.write(pack_uint32(self.unknown_14))
        file.write(pack_uint32(self.unknown_15))
        file.write(pack_uint32(self.unknown_16))
        self.unknown_quaternion.write(file)

    def read(self, file):
        pass

    def size(self):
        return 16 * CONST_UINT32_SIZE + self.unknown_quaternion.size()

class Cas2Header:
    CONST_CS2_FILE_TAG = bytearray([0x01, 0x32, 0x53, 0x43])
    CONST_EXPORTER_VERSION = "Unofficial Cas2 Exporter v1.0 by Mr.Jox aka victimized."
    
    def __init__(self, file_info):
        self.header_size = 0
        self.unknown_1 = 0
        self.unknown_2 = 0.0
        self.addon_details = ""
        self.file_details = Utf16String(file_info)
        
    def write(self, file):        
        self.unknown_2 = 1.16 # Unknown but looks like a file version
        self.addon_details = Utf16String("Blender " + bpy.app.version_string + ". " + self.CONST_EXPORTER_VERSION)
        
        unknown_1_data = pack_uint32(self.unknown_1)
        unknown_2_data = pack_float(self.unknown_2)
        
        self.header_size += 4 # size of self.header_size
        self.header_size += len(unknown_1_data)
        self.header_size += len(unknown_2_data)
        self.header_size += self.addon_details.size()
        self.header_size += self.file_details.size()
        
        file.write(self.CONST_CS2_FILE_TAG)
        file.write(pack_uint32(self.header_size))
        file.write(unknown_1_data)
        file.write(unknown_2_data)
        self.addon_details.write(file)
        self.file_details.write(file)
    
    def read(self, file):
        file_tag = file.read(4)
        if file_tag != self.CONST_CS2_FILE_TAG:
            print("Failed - The file i not a .CS2 file.")
        else:
            print("OK - The file is a .CS2 file.")

    def size(self):
        return self.header_size + 4

class SceneInfoBlock:
    def __init__(self):
        self.block_size = 0
        self.unknown_1 = 0
        self.scene_object_types_count = 0
        self.unknown_2 = 0
        self.cameras_count = 0
        self.rigid_models_count = 0
        self.unknown_3 = 0
        self.unknown_4 = 0
        self.weighted_models_count = 0
        self.lines_count = 0
        self.dummies_count = 0
        self.materials_count = 0
        self.unknown_5 = 0
        self.unknown_6 = 0
        self.instances_count = 0

    def write(self, file):
        self.unknown_1 = 1 # unknown but likely an array size
        self.scene_object_types_count = 23 # total number of all scene object types (camera, line, etc)
        self.block_size = 12 + 4 * self.scene_object_types_count # block_size itself + unknown_1 + scene_object_types_count + 4 * scene_object_types_count

        file.write(pack_uint32(self.block_size))
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.scene_object_types_count))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.cameras_count))
        file.write(pack_uint32(self.rigid_models_count))
        file.write(pack_uint32(self.unknown_3))
        file.write(pack_uint32(self.unknown_4))
        file.write(pack_uint32(self.weighted_models_count))
        file.write(pack_uint32(self.lines_count))
        file.write(pack_uint32(self.dummies_count))
        file.write(pack_uint32(self.materials_count))
        file.write(pack_uint32(self.unknown_5))
        file.write(pack_uint32(self.unknown_6))
        file.write(pack_uint32(self.instances_count))
        file.write(bytearray(11 * CONST_UINT32_SIZE)) # 11 unknown uint32 values, likely just reserved for future use

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class KeyFramesBlock1:
    def __init__(self):
        self.block_size = 0
        self.unknown_1 = 0
        self.key_start = 0.0
        self.key_end = 0.0

    def write(self, file):
        self.unknown_1 = 2 # No clue whatsoever

        unknown_struct_1 = UnknownKeyFramesBlockStruct1()
        unknown_struct_2 = UnknownKeyFramesBlockStruct2()

        unknown_struct_1.fill_temp_data()
        unknown_struct_2.fill_temp_data()

        self.block_size = 16 # block_size itself + unknown_1 + key_start + key_end
        self.block_size += unknown_struct_1.size()
        self.block_size += unknown_struct_2.size()

        file.write(pack_uint32(self.block_size))
        file.write(pack_uint32(self.unknown_1))
        file.write(pack_float(self.key_start))
        file.write(pack_float(self.key_end))
        unknown_struct_1.write(file)
        unknown_struct_2.write(file)

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class KeyFramesBlock2:
    def __init__(self):
        self.block_size = 0
        self.unknown_1 = 0

    def write(self, file):
        self.unknown_1 = 22 # Unknown but always appears to be 22
        self.block_size = 8 # block_size + unknown_1

        unknown_struct_1 = UnknownKeyFramesBlockStruct3()
        unknown_struct_1.fill_temp_data()

        self.block_size += unknown_struct_1.size()

        file.write(pack_uint32(self.block_size))
        file.write(pack_uint32(self.unknown_1))
        unknown_struct_1.write(file)

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class VertexDataRigidModel:
    def __init__(self, position, normal, color, tex_coords):
        self.block_size = 0
        self.position = position
        self.normal = normal
        self.color = color
        self.tex_coords = tex_coords
        self.unknown = 0.0

    def write(self, file):
        self.block_size += self.position.size()
        self.block_size += self.normal.size()
        self.block_size += self.color.size()
        self.block_size += CONST_UINT32_SIZE + CONST_FLOAT_SIZE # len(self.tex_coords) + self.unknown

        self.position.write(file)
        self.normal.write(file)
        self.color.write(file)

        file.write(pack_uint32(len(self.tex_coords)))
        for uvw_coord in self.tex_coords:
            self.block_size += uvw_coord.size()
            uvw_coord.write(file)

        file.write(pack_float(self.unknown))

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class MeshTriangle:
    def __init__(self, index_1, index_2, index_3):
        self.vertex_index_1 = index_1
        self.vertex_index_2 = index_2
        self.vertex_index_3 = index_3

    def write(self, file):
        file.write(pack_uint32(self.vertex_index_1))
        file.write(pack_uint32(self.vertex_index_2))
        file.write(pack_uint32(self.vertex_index_3))

    def read(self, file):
        pass

    def size(self):
        return 3 * CONST_UINT32_SIZE

class SubMesh:
    def __init__(self, material_index, triangles):
        self.block_size = 0
        self.triangles = triangles
        self.materialId = material_index

    def write(self, file):
        self.block_size += CONST_UINT32_SIZE + CONST_INT32_SIZE # materialId + len(triangles)

        file.write(pack_uint32(len(self.triangles)))

        for triangle in self.triangles:
            self.block_size += triangle.size()
            triangle.write(file)

        file.write(pack_int32(self.materialId))

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class BoundingBox:
    def __init__(self, obj):
        self.min_x = CONST_FLOAT_MAX
        self.min_y = CONST_FLOAT_MAX
        self.min_z = CONST_FLOAT_MAX
        self.max_x = CONST_FLOAT_MIN
        self.max_y = CONST_FLOAT_MIN
        self.max_z = CONST_FLOAT_MIN

        for corner in obj.bound_box:
            if self.min_x > corner[0]:
                self.min_x = corner[0]

            if self.min_y > corner[1]:
                self.min_y = corner[1]

            if self.min_z > corner[2]:
                self.min_z = corner[2]

            if self.max_x < corner[0]:
                self.max_x = corner[0]

            if self.max_y < corner[1]:
                self.max_y = corner[1]

            if self.max_z < corner[2]:
                self.max_z = corner[2]

    def write(self, file):
        file.write(pack_float(self.min_x))
        file.write(pack_float(self.min_z))
        file.write(pack_float(self.min_y))
        file.write(pack_float(self.max_x))
        file.write(pack_float(self.max_z))
        file.write(pack_float(self.max_y))

    def read(self, file):
        pass

    def size(self):
        return 6 * CONST_FLOAT_SIZE

class GeometryDataHeader:
    def __init__(self, obj):
        self.block_size = 0
        self.unknown_1 = 11
        self.unknown_2 = 0
        self.unknown_3 = 0
        self.unknown_float_array = [0.0]
        self.bounding_box_array = [BoundingBox(obj)]
        self.geometry_data_type = GeometryDataType.UNSET

    def write(self, file):
        self.block_size += 4 + 4 + 4 # unknown_1 + unknown_2 + unknown_3
        self.block_size += len(self.unknown_float_array) * 4
        self.block_size += len(self.bounding_box_array)
        self.block_size += 4 # GeometryDataType

        file.write(pack_uint32(self.unknown_1))
        file.write(pack_uint32(self.unknown_2))
        file.write(pack_uint32(self.unknown_3))

        file.write(pack_uint32(len(self.unknown_float_array)))
        for value in self.unknown_float_array:
            file.write(pack_float(value))

        file.write(pack_uint32(len(self.bounding_box_array)))
        for bbox in self.bounding_box_array:
            self.block_size += bbox.size()
            bbox.write(file)

        file.write(pack_int32(self.geometry_data_type.value))

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class GeometryDataMesh:
    def __init__(self, obj):
        self.header = GeometryDataHeader(obj)
        self.header.geometry_data_type = GeometryDataType.MESH
        self.block_size = self.header.size()

        self.uv_channels = {}
        for index, uv_layer in enumerate(obj.data.uv_layers):
            self.uv_channels[uv_layer.name] = index

        self.vertices = []
        self.sub_meshes = []
        fill_mesh_data(self, obj, False)
        self.unknown_related_to_vertex_colors = 0
    
    def write(self, file):
        self.header.write(file)
        self.block_size += CONST_UINT32_SIZE + CONST_UINT32_SIZE + CONST_UINT32_SIZE + CONST_UINT32_SIZE # len(uv_channels) + len(vertices) + len(sub_meshes) + unknown_related_to_vertex_colors

        file.write(pack_uint32(len(self.uv_channels)))
        for uv_layer in self.uv_channels:
            self.block_size += CONST_UINT32_SIZE
            file.write(pack_uint32(self.uv_channels[uv_layer]))

        file.write(pack_uint32(len(self.vertices)))
        for vert in self.vertices:
            self.block_size += vert.size()
            vert.write(file)

        file.write(pack_uint32(len(self.sub_meshes)))
        for submesh in self.sub_meshes:
            self.block_size += submesh.size()
            submesh.write(file)

        file.write(pack_uint32(self.unknown_related_to_vertex_colors))

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class GeometryDataLine:
    def __init__(self, obj):
        self.header = GeometryDataHeader(obj)
        self.header.geometry_data_type = GeometryDataType.LINE
        self.block_size = self.header.size()

    def write(self, file):
        self.header.write(file)

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class ModelNode:
    def __init__(self, obj):
        self.block_size = 0
        self.node_type = obj_to_node_type(obj)
        self.node_name = Utf16String(obj.name)
        self.unknown_string = Utf16String() # Always empty?
        self.user_defined_properties = Utf16String(build_properties(obj))
        self.node_index = 0
        self.attributes = NodeAttributes(obj)
        self.geometry_data = [GeometryDataMesh(obj)]

    def write(self, file):
        self.block_size = 8 # block_size + node_type
        self.block_size += self.node_name.size()
        self.block_size += self.unknown_string.size()
        self.block_size += self.user_defined_properties.size()
        self.block_size += 4 # node_index
        self.block_size += self.attributes.size()
        self.block_size += 4 * len(self.geometry_data)

        file.write(pack_uint32(self.block_size))
        file.write(pack_uint32(self.node_type.value))
        self.node_name.write(file)
        self.unknown_string.write(file)
        self.user_defined_properties.write(file)
        file.write(pack_uint32(self.node_index))
        self.attributes.write(file)

        file.write(pack_uint32(len(self.geometry_data)))
        for geometry in self.geometry_data:
            self.block_size += geometry.size()
            geometry.write(file)

    def read(self, file):
        pass
    
    def size(self):
        return self.block_size

class SceneNode:
    def __init__(self):
        pass

    def write(self, file):
        pass

    def read(self, file):
        pass
    
    def size(self):
        pass

class SceneRootBlock:
    def __init__(self, file_info):
        self.block_size = 0
        self.node_type = NodeType.SCENE_ROOT
        self.nodes_count = 0
        self.node_name = Utf16String("Exporter Inserted Root")
        self.unknown_1 = -1
        self.unknown_2 = 1
        self.unknown_data_1 = UnknownSceneRootData1()
        self.unknown_3 = 0
        self.file_info = Utf16String(file_info)
        self.unknown_4 = 0
        self.unknown_5 = 0
        self.unknown_6 = 0
        self.unknown_7 = 0
        self.unknown_8 = 0

    def write(self, file):
        self.block_size = 20 # block_size + node_type + nodes_count + unknown_1 + unknown_2
        self.block_size += self.unknown_data_1.size() + 4 # unknown_data_1 + unknown_3
        self.block_size += self.file_info.size()
        self.block_size += 5 * CONST_UINT32_SIZE # unknown_4 ... unknown_8

        #self.nodes_count =  + 1 # +1 is the Exporter Inserted Root

        #for i in range():
        #    self.block_size += .size()

        file.write(pack_uint32(self.block_size))
        file.write(pack_uint32(self.node_type.value))
        file.write(pack_uint32(self.nodes_count))
        self.node_name.write(file)
        file.write(pack_int32(self.unknown_1))
        file.write(pack_uint32(self.unknown_2))
        self.unknown_data_1.write(file)
        file.write(pack_uint32(self.unknown_3))
        self.file_info.write(file)
        file.write(bytearray(5 * CONST_UINT32_SIZE))

    def read(self, file):
        pass

    def size(self):
        return self.block_size

class ExportCas2File(bpy.types.Operator):
    bl_idname = "export_scene.cas2_file"
    bl_label = "Export Cas2 file (.cs2)"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        export_filename = self.filepath + ".cs2"
        with open(export_filename, 'wb') as file:
            file_info = build_file_info(export_filename)

            header = Cas2Header(file_info)
            scene_info = SceneInfoBlock()
            key_frames_1_info = KeyFramesBlock1()
            key_frames_2_info = KeyFramesBlock2()
            scene_root = SceneRootBlock(file_info)

            rigid_models = create_rigid_models()
            scene_info.rigid_models_count = len(rigid_models)

            header.write(file)
            scene_info.write(file)
            key_frames_1_info.write(file)
            key_frames_2_info.write(file)

            for model in rigid_models:
                model.write(file)

            scene_root.write(file)
        
        unregister()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def pack_uint16(value):
    return struct.pack('H', value)

def pack_int16(value):
    return struct.pack('h', value)

def pack_uint32(value):
    return struct.pack('I', value)

def pack_int32(value):
    return struct.pack('i', value)

def pack_float(value):
    return struct.pack('f', value)

def property_value_to_string(value):
    if isinstance(value, (int, float, bool)):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        if len(value) == 1:
            return str(value[0])
        elif len(value) == 2:
            return str(f"x:\"{value[0]}\" y:\"{value[1]}\"")
        elif len(value) == 3:
            return str(f"x:\"{value[0]}\" y:\"{value[1]}\" z:\"{value[2]}\"")
        elif len(value) == 4:
            return str(f"x:\"{value[0]}\" y:\"{value[1]}\" z:\"{value[2]}\" w:\"{value[3]}\"")
    
    return str(value)

def build_properties(obj):
    if obj.data is None:
        return ""

    properties = ""

    for prop_name, prop_value in obj.items():
        if prop_name.startswith(CONST_OBJECT_PROPERTY_PREFIX):
            properties += prop_name[len(CONST_OBJECT_PROPERTY_PREFIX):] + " = " + property_value_to_string(prop_value) + "\n"

    return properties

def build_file_info(filepath):
    blend_file_name = bpy.data.filepath
    if not blend_file_name:
        blend_file_name = "unsaved"
    
    info_username = "USERNAME:\t" + os.getlogin()
    info_export_data = "EXPORTED:\t" + datetime.now().strftime('%d/%m/%Y,%H:%M:%S')
    info_cas_name = "CAS_NAME:\t" + filepath
    info_blend_file = "BLEND_FILE:\t" + blend_file_name
    return info_username + "\n" + info_export_data + "\n" + info_cas_name + "\n" + info_blend_file

def build_attributes(obj):
    if obj.data is None:
        return {}

    attributes = {
        CONST_ATTRIBUTES_STRING: [],
        CONST_ATTRIBUTES_INT: [],
        CONST_ATTRIBUTES_FLOAT: [],
        CONST_ATTRIBUTES_VEC3: [],
        CONST_ATTRIBUTES_VEC4: [],
    }

    for prop_name, prop_value in obj.items():
        if prop_name.startswith(CONST_OBJECT_ATTRIBUTE_PREFIX) == False:
            continue

        attribute_name = prop_name[len(CONST_OBJECT_ATTRIBUTE_PREFIX):]

        if isinstance(prop_value, bool):
            int_value = 1 if prop_value == True else 0
            attributes[CONST_ATTRIBUTES_INT].append(IntAttribute(attribute_name, int_value))
        elif isinstance(prop_value, int):
            attributes[CONST_ATTRIBUTES_INT].append(IntAttribute(attribute_name, prop_value))
        elif isinstance(prop_value, float):
            attributes[CONST_ATTRIBUTES_FLOAT].append(FloatAttribute(attribute_name, prop_value))
        elif isinstance(prop_value, str):
            attributes[CONST_ATTRIBUTES_STRING].append(StringAttribute(attribute_name, prop_value))
        elif len(value) == 3:
            attributes[CONST_ATTRIBUTES_VEC3].append(Vec3Attribute(attribute_name, Vec3(prop_value[0], prop_value[1], prop_value[2])))
        elif len(value) == 4:
            attributes[CONST_ATTRIBUTES_VEC4].append(Vec4Attribute(attribute_name, Vec3(prop_value[0], prop_value[1], prop_value[2], prop_value[3])))

    return attributes

def is_weighted_model(obj):
    if obj.type != 'MESH':
        return False

    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            return 

    return False

def is_rigid_model(obj):
    if obj.type != 'MESH':
        return False

    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            return False

    return True

def create_rigid_models():
    models = []

    for obj in bpy.data.objects:
        if is_rigid_model(obj):
            model = ModelNode(obj)
            models.append(model)

    return models

def obj_to_node_type(obj):
    if is_rigid_model(obj):
        return NodeType.RIGID_MODEL

    if is_weighted_model(obj):
        return NodeType.WEIGHTED_MODEL

def triangulate_mesh(obj):
    mesh_copy = obj.data.copy()

    bm = bmesh.new()
    bm.from_mesh(mesh_copy)

    # Swap Y and Z axes
    for v in bm.verts:
        v.co.y, v.co.z = v.co.z, v.co.y
        v.normal.y, v.normal.z = v.normal.z, v.normal.y
    
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bpy.data.meshes.remove(mesh_copy)

    bm.normal_update()

    return bm

def fill_mesh_data(geometry_data, obj, fill_skin_data):
    bm = triangulate_mesh(obj)

    color_layer_index = -1
    if len(bm.loops.layers.color) > 0:
        color_layer_index = 0

    uv_layers = bm.loops.layers.uv.values()

    num_materials = len(obj.data.materials)
    
    sub_meshes = {}

    for i in range(0, num_materials):
        sub_meshes[i] = []

    if num_materials == 0:
        sub_meshes[0] = []

    vert_index = 0

    for face in bm.faces:
        assert len(face.verts) == 3, f"Face {face.index} is not a triangle."
        triangle_indices = []

        face.normal_flip()
        for loop in face.loops:
            vert = loop.vert

            position = Vec3(vert.co.x, vert.co.y, vert.co.z)
            normal = Vec3(face.normal.x, face.normal.y, face.normal.z)

            if color_layer_index != -1:
                color_layer = bm.loops.layers.color[color_layer_index]
                color = Vec4(loop[color_layer].x, loop[color_layer].y, loop[color_layer].z, loop[color_layer].w)
            else:
                color = Vec4()

            uvs = []
            for uv_layer in uv_layers:
                uv = Vec3(loop[uv_layer].uv.x, loop[uv_layer].uv.y)
                uvs.append(uv)

            vertexData = VertexDataRigidModel(position, normal, color, uvs)
            geometry_data.vertices.append(vertexData)
            triangle_indices.append(vert_index)

            vert_index = vert_index + 1

        triangle = MeshTriangle(triangle_indices[0], triangle_indices[1], triangle_indices[2])
        sub_meshes[0 if num_materials == 0 else face.material_index].append(triangle)

    for i in range(0, len(sub_meshes)):
        material_index = -1 if num_materials == 0 else i
        submesh_data = SubMesh(material_index, sub_meshes[i])
        geometry_data.sub_meshes.append(submesh_data)

    bm.free()

def get_color_for_vertex(bm, vertex_index):
    if len(bm.loops.layers.color) > 0:
        color_layer = bm.loops.layers.color[0]

        for face in bm.faces:
            for loop in face.loops:
                if loop.vert.index != vertex_index:
                    continue

                vertex_color = loop[color_layer]
                return Vec4(vertex_color.x, vertex_color.y, vertex_color.z, vertex_color.w)

    return Vec4(0.0, 0.0, 0.0, 0.0)

def get_uvw_for_vertex(bm, vertex_index):
    tex_coords = []

    for uv_layer in bm.loops.layers.uv:
        for face in bm.faces:
            for loop in face.loops:
                if loop.vert.index != vertex_index:
                    continue

                uv_coords = loop[uv_layer].uv
                tex_coords.append(Vec3(uv_coords.x, uv_coords.y))

    return tex_coords

def menu_func_export(self, context):
    self.layout.operator(ExportCas2File.bl_idname, text="Export Cas2 (.cs2)")

def register():
    bpy.utils.register_class(ExportCas2File)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ExportCas2File)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
