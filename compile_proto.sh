#!/usr/bin/env bash

protoc --proto_path=. --python_out=. ./protos/*.proto