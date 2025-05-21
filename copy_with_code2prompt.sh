#!/bin/bash

# Create a temporary directory
tmp_dir=$(mktemp -d)
echo "Created temporary directory: $tmp_dir"

# # Copy all the query files to the temp directory
# cp get_logs_final_query.txt "$tmp_dir/"
# cp get_logs_final_query_slow.txt "$tmp_dir/"
# cp get_logs_metric_final_query.txt "$tmp_dir/"
# cp get_logs_metric_final
# _query_slow.txt "$tmp_dir/"

# # Extract the functions from helpers.py into separate files
# echo "Extracting functions from helpers.py..."
#grep -A 500 "def create_logs" orchestra/web/api/log/views.py | grep -B 500 -m 2 "def " > "$tmp_dir/create_logs.py"
#grep -A 500 "def update_logs" orchestra/web/api/log/views.py | grep -B 500 -m 2 "def " > "$tmp_dir/update_logs.py"
# grep -A 500 "def _build_subquery_for_identifier" orchestra/web/api/log/helpers.py | grep -B 500 -m 2 "def " > "$tmp_dir/_build_subquery_for_identifier.py"
# grep -A 1000 "def _handle_functions" orchestra/web/api/log/helpers.py | grep -B 1000 -m 2 "def " > "$tmp_dir/_handle_functions.py"

# Copy the full files to the temp directory
cp -r unity/knowledge_manager "$tmp_dir/knowledge_manager"
cp -r unity/communication "$tmp_dir/communication"
cp -r unity/task_list_manager "$tmp_dir/task_list_manager"
cp -r unity/task_manager "$tmp_dir/task_manager"
cp -r unity/events "$tmp_dir/events"
cp -r unity/common "$tmp_dir/common"
cp -r sandboxes "$tmp_dir/sandboxes"
cp -r tests "$tmp_dir/tests"


#cp orchestra/workers/vector_worker.py "$tmp_dir/vector_worker.py"
#cp orchestra/tests/test_log/test_log_filtering.py "$tmp_dir/test_log_filtering.py"
#cp orchestra/tests/test_log/test_log_derived.py "$tmp_dir/test_log_derived.py"
#cp orchestra/conftest.py "$tmp_dir/conftest.py"
#cp orchestra/settings.py "$tmp_dir/settings.py"
#cp sss_orchestra.py "$tmp_dir/"

# Change to the temp directory
cd "$tmp_dir"

# Use code2prompt to copy all files in the directory
echo "Copying all files to clipboard using code2prompt..."
code2prompt --exclude "*.pyc" $tmp_dir --exclude-from-tree

# now delete the temp directory
rm -rf $tmp_dir
echo "Deleted temporary directory: $tmp_dir"
# Return to the original directory
cd -

echo "All files copied to clipboard! Temporary directory: $tmp_dir"
