allow_list=$(< "${ALLOWED_DEPENDENCY_FILEPATH}" grep -oE "^[a-zA-Z0-9]+[a-zA-Z0-9_\-]*" | paste -d "|" -s -)
cleaned_allow_list=$(echo $allow_list | sed "s/keras_preprocessing/Keras-Preprocessing/")
echo "Allow list: ${cleaned_allow_list}"
pip freeze | grep -v "facilyst.git" | grep -E "${cleaned_allow_list}" > "${LATEST_DEPENDENCY_FILEPATH}"