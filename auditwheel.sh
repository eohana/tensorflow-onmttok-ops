#!/usr/bin/env bash
TF_SHARED_LIBRARY_NAME=\"libtensorflow_framework.so.2\"
POLICY_JSON=$(find / -name policy.json)

sed -i "s/libresolv.so.2\"/libresolv.so.2\", $TF_SHARED_LIBRARY_NAME/g" $POLICY_JSON
cat $POLICY_JSON

auditwheel $@
