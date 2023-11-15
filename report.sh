#!/bin/bash
apt-get update;
apt-get install -y locales;
echo "en_US.UTF-8 UTF-8" > /etc/locale.gen;
/usr/sbin/locale-gen;
pip install pyyaml --break-system-packages --root-user-action=ignore;
cd ./report;
make clean;
make;
