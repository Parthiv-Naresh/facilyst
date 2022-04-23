allow_list=$(< requirements.txt grep -oE "^[a-zA-Z0-9]+[a-zA-Z0-9_\-]*" | paste -d "|" -s -)
echo "Allow list: ${allow_list}"
pip freeze | grep -v "facilyst.git" | grep -E "${allow_list}" > "${LATEST_DEPENDENCY_FILEPATH}"