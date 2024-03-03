#!/bin/bash
#
# ./examples/ts-type-to-grammar.sh "{a:string,b:string,c?:string}"
# python examples/json-schema-to-grammar.py https://raw.githubusercontent.com/SchemaStore/schemastore/master/src/schemas/json/tsconfig.json
#
set -euo pipefail

readonly type="$1"

# Create a temporary directory
TMPDIR=""
trap 'rm -fR "$TMPDIR"' EXIT
TMPDIR=$(mktemp -d)

DTS_FILE="$TMPDIR/type.d.ts"
SCHEMA_FILE="$TMPDIR/schema.json"

echo "export type MyType = $type" > "$DTS_FILE"

# https://github.com/YousefED/typescript-json-schema
# npx typescript-json-schema --defaultProps --required "$DTS_FILE" MyType | tee "$SCHEMA_FILE" >&2

# https://github.com/vega/ts-json-schema-generator
npx ts-json-schema-generator --path "$DTS_FILE" --type MyType -e none | tee "$SCHEMA_FILE" >&2

./examples/json-schema-to-grammar.py "$SCHEMA_FILE"