#!/bin/bash

# 找到当前目录及子目录中大于 20MB 的文件
find . -type f -size +20M -not -path "./.git/*" -exec echo {} >> .gitignore \;

# 提示用户已将文件添加到.gitignore
echo "Files larger than 20MB have been added to .gitignore."