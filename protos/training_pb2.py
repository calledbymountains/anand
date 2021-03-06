# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/training.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import data_pb2 as protos_dot_data__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/training.proto',
  package='protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x15protos/training.proto\x12\x06protos\x1a\x11protos/data.proto\"\x8e\x02\n\ttraineval\x12\x0f\n\x07numgpus\x18\x01 \x02(\x05\x12\x18\n\x0etraining_steps\x18\x80\xf1\x04 \x02(\x05\x12\x1c\n\x10\x65val_after_steps\x18\x03 \x02(\x05:\x02\x31\x30\x12\x1c\n\rlearning_rate\x18\x04 \x01(\x02:\x05\x30.001\x12\x19\n\rdisplay_steps\x18\n \x01(\x05:\x02\x31\x30\x12\x15\n\x08momentum\x18\x05 \x01(\x02:\x03\x30.9\x12\x1f\n\tdata_info\x18\x06 \x02(\x0b\x32\x0c.protos.data\x12\x12\n\nnumclasses\x18\x07 \x02(\x05\x12\x10\n\x08\x62\x61sepath\x18\x08 \x02(\t\x12!\n\x14save_checkpoint_secs\x18\t \x01(\x05:\x03\x31\x32\x30\"/\n\x08Training\x12#\n\x08training\x18\x01 \x02(\x0b\x32\x11.protos.traineval')
  ,
  dependencies=[protos_dot_data__pb2.DESCRIPTOR,])




_TRAINEVAL = _descriptor.Descriptor(
  name='traineval',
  full_name='protos.traineval',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='numgpus', full_name='protos.traineval.numgpus', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='training_steps', full_name='protos.traineval.training_steps', index=1,
      number=80000, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eval_after_steps', full_name='protos.traineval.eval_after_steps', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='protos.traineval.learning_rate', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.001),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='display_steps', full_name='protos.traineval.display_steps', index=4,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='protos.traineval.momentum', index=5,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.9),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_info', full_name='protos.traineval.data_info', index=6,
      number=6, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='numclasses', full_name='protos.traineval.numclasses', index=7,
      number=7, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basepath', full_name='protos.traineval.basepath', index=8,
      number=8, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_checkpoint_secs', full_name='protos.traineval.save_checkpoint_secs', index=9,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=120,
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
  serialized_start=53,
  serialized_end=323,
)


_TRAINING = _descriptor.Descriptor(
  name='Training',
  full_name='protos.Training',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='training', full_name='protos.Training.training', index=0,
      number=1, type=11, cpp_type=10, label=2,
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
  serialized_start=325,
  serialized_end=372,
)

_TRAINEVAL.fields_by_name['data_info'].message_type = protos_dot_data__pb2._DATA
_TRAINING.fields_by_name['training'].message_type = _TRAINEVAL
DESCRIPTOR.message_types_by_name['traineval'] = _TRAINEVAL
DESCRIPTOR.message_types_by_name['Training'] = _TRAINING
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

traineval = _reflection.GeneratedProtocolMessageType('traineval', (_message.Message,), dict(
  DESCRIPTOR = _TRAINEVAL,
  __module__ = 'protos.training_pb2'
  # @@protoc_insertion_point(class_scope:protos.traineval)
  ))
_sym_db.RegisterMessage(traineval)

Training = _reflection.GeneratedProtocolMessageType('Training', (_message.Message,), dict(
  DESCRIPTOR = _TRAINING,
  __module__ = 'protos.training_pb2'
  # @@protoc_insertion_point(class_scope:protos.Training)
  ))
_sym_db.RegisterMessage(Training)


# @@protoc_insertion_point(module_scope)
