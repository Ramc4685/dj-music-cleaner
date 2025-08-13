#!/bin/bash
# Rekordbox Round-Trip Validation Runner
# This script automates the entire validation process for Rekordbox XML round-trip preservation

set -e  # Exit on any error

# Configuration - adjust these paths as needed
TEST_MUSIC_DIR="${1:-./test_music}"
OUTPUT_DIR="${2:-./output}"
REKORDBOX_XML="${3:-./rekordbox.xml}"
REPORTS_DIR="${4:-./reports}"
VALIDATION_REPORT="${REPORTS_DIR}/rekordbox_validation_report.txt"

# Print header
echo "=========================================================="
echo "REKORDBOX ROUND-TRIP VALIDATION TEST"
echo "=========================================================="
echo "Test music directory: ${TEST_MUSIC_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Rekordbox XML: ${REKORDBOX_XML}"
echo "Reports directory: ${REPORTS_DIR}"
echo "=========================================================="

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPORTS_DIR}"

# Step 1: Run DJ Music Cleaner with rekordbox-preserve option
echo "Step 1: Processing test files with DJ Music Cleaner..."
python -m djmusiccleaner.dj_music_cleaner \
  --input "${TEST_MUSIC_DIR}" \
  --output "${OUTPUT_DIR}" \
  --rekordbox "${REKORDBOX_XML}" \
  --rekordbox-preserve \
  --export-xml \
  --report

# Step 2: Locate the generated XML file (it should be in the output directory)
GENERATED_XML=$(find "${OUTPUT_DIR}" -name "*.xml" -type f -print -quit)

if [ -z "${GENERATED_XML}" ]; then
  echo "Error: No XML file was generated in the output directory"
  exit 1
fi

echo "Generated XML found at: ${GENERATED_XML}"

# Step 3: Run the validation script
echo "Step 3: Running XML validation..."
python tests/rekordbox/validate_xml.py \
  --original "${REKORDBOX_XML}" \
  --processed "${GENERATED_XML}" \
  --report "${VALIDATION_REPORT}"

# Step 4: Check validation result
if [ $? -eq 0 ]; then
  echo "Validation PASSED! Report saved to ${VALIDATION_REPORT}"
  exit 0
else
  echo "Validation FAILED! Review the report at ${VALIDATION_REPORT}"
  exit 1
fi
