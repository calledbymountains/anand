# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/data.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/data.proto',
  package='protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x11protos/data.proto\x12\x06protos\"\x91\x02\n\ndata_proto\x12\x1a\n\x12tfrecord_list_glob\x18\x01 \x02(\t\x12\x1a\n\x12num_parallel_reads\x18\x02 \x02(\x05\x12\x11\n\tcache_dir\x18\x03 \x02(\t\x12\x16\n\nbatch_size\x18\x04 \x02(\x05:\x02\x33\x32\x12!\n\x14prefetch_buffer_size\x18\x05 \x02(\x05:\x03\x31\x30\x30\x12#\n\x16map_num_parallel_calls\x18\x06 \x02(\x05:\x03\x31\x30\x30\x12\x16\n\x07shuffle\x18\x07 \x02(\x08:\x05\x66\x61lse\x12 \n\x13shuffle_buffer_size\x18\x08 \x02(\x05:\x03\x31\x30\x30\x12\x1e\n\x10read_buffer_size\x18\t \x02(\x05:\x04\x31\x30\x30\x30\"T\n\x04\x64\x61ta\x12&\n\ntrain_data\x18\x01 \x02(\x0b\x32\x12.protos.data_proto\x12$\n\x08val_data\x18\x02 \x02(\x0b\x32\x12.protos.data_proto')
)




_DATA_PROTO = _descriptor.Descriptor(
  name='data_proto',
  full_name='protos.data_proto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tfrecord_list_glob', full_name='protos.data_proto.tfrecord_list_glob', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_parallel_reads', full_name='protos.data_proto.num_parallel_reads', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cache_dir', full_name='protos.data_proto.cache_dir', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='protos.data_proto.batch_size', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=32,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prefetch_buffer_size', full_name='protos.data_proto.prefetch_buffer_size', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='map_num_parallel_calls', full_name='protos.data_proto.map_num_parallel_calls', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shuffle', full_name='protos.data_proto.shuffle', index=6,
      number=7, type=8, cpp_type=7, label=2,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shuffle_buffer_size', full_name='protos.data_proto.shuffle_buffer_size', index=7,
      number=8, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='read_buffer_size', full_name='protos.data_proto.read_buffer_size', index=8,
      number=9, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=1000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=303,
)


_DATA = _descriptor.Descriptor(
  name='data',
  full_name='protos.data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='train_data', full_name='protos.data.train_data', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='val_data', full_name='protos.data.val_data', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=305,
  serialized_end=389,
)

_DATA.fields_by_name['train_data'].message_type = _DATA_PROTO
_DATA.fields_by_name['val_data'].message_type = _DATA_PROTO
DESCRIPTOR.message_types_by_name['data_proto'] = _DATA_PROTO
DESCRIPTOR.message_types_by_name['data'] = _DATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

data_proto = _reflection.GeneratedProtocolMessageType('data_proto', (_message.Message,), dict(
  DESCRIPTOR = _DATA_PROTO,
  __module__ = 'protos.data_pb2'
  # @@protoc_insertion_point(class_scope:protos.data_proto)
  ))
_sym_db.RegisterMessage(data_proto)

data = _reflection.GeneratedProtocolMessageType('data', (_message.Message,), dict(
  DESCRIPTOR = _DATA,
  __module__ = 'protos.data_pb2'
  # @@protoc_insertion_point(class_scope:protos.data)
  ))
_sym_db.RegisterMessage(data)


# @@protoc_insertion_point(module_scope)
