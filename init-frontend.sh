#!/usr/bin/env bash
set -e
REPO_ROOT=$(pwd)

# 创建子目录
mkdir -p frontend
cd frontend

# 生成最小 package.json
cat > package.json <<'EOF'
{
  "name": "frontend",
  "version": "1.0.0",
  "scripts": {
    "build": "mkdir -p dist && cp index.html dist/"
  },
  "devDependencies": {}
}
EOF

# 生成示例页面
cat > index.html <<'EOF'
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Stock Analysis</title>
</head>
<body>
  <h1>Hello GitHub Pages!</h1>
  <p>Deployed via GitHub Actions.</p>
</body>
</html>
EOF

# 回到仓库根
cd "$REPO_ROOT"

# 首次提交
git add .
git commit -m "feat: add frontend subdir with package.json"
git push origin main
echo ">>> 推送完成，GitHub Actions 会自动开始部署 ..."
