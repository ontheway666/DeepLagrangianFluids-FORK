# 复制一个文件中的所有py文件到另一个目录，保持相对路径

# 如果目标路径有多余的文件，不会删除

#   容器  》 Sync   的同步

# sync中内容修改的概率非常小，所以可以认为image中的才是最新的，强制覆盖（不加-u）问题也不大


rsync -uvrpt --include='*/' --include='*.py' --include='*.sh' --include='*.h5' \
       	--include='*.json' --include='*.yaml' \
		--include='*.obj'  --include='*.npy' \
       	--include='./scripts/cache'       /w/bkpCode927/yzx3/ ./



# --exclude='*'