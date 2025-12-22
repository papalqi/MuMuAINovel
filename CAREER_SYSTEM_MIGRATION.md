# 职业系统迁移指南

## 📋 概述

该指南适用于升级到包含职业系统（Career System）功能的版本的用户。职业系统引入了两个新的数据库表和对现有 `characters` 表的扩展。

## 🔍 适用场景

### 情况1：新用户部署（推荐）
**条件**：首次部署MuMuAINovel

**操作**：无需额外步骤
- 应用启动时会自动创建所有必需的表和列
- SQLAlchemy ORM会在首次初始化时建立完整的数据库架构

### 情况2：现有用户升级（需要操作）
**条件**：已有运行中的MuMuAINovel实例，需要升级到职业系统版本

**需要执行**：数据库迁移脚本

---

## 🛠️ 升级步骤

### 步骤1：备份数据库
```bash
# 使用Docker备份PostgreSQL数据库
docker exec mumuainovel-postgres pg_dump -U mumuai mumuai_novel > backup_before_career_migration.sql

# 或使用Docker Compose卷备份
docker run -v mumuainovel_postgres_data:/data -v $(pwd):/backup \
  postgres:18-alpine \
  tar czf /backup/postgres_backup_$(date +%Y%m%d_%H%M%S).tar.gz /data
```

### 步骤2：升级代码
```bash
# 拉取最新代码
git pull origin main

# 如果使用Docker
docker-compose down
docker-compose up -d  # 重新启动带最新代码的容器
```

### 步骤3：执行数据库迁移
在升级后首次启动应用之前执行迁移脚本：

```bash
# 方式A：使用Docker exec执行（推荐）
docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel \
  -f /scripts/create_career_tables.sql

# 方式B：使用本地psql（如果已安装PostgreSQL）
psql -h localhost -U mumuai -d mumuai_novel \
  -f backend/scripts/create_career_tables.sql
```

### 步骤4：验证迁移
```bash
# 检查新表是否创建
docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel -c "
  SELECT table_name FROM information_schema.tables 
  WHERE table_schema = 'public' 
  ORDER BY table_name;"

# 检查characters表的新列
docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel -c "
  \d characters"
```

**预期结果：**
- 应看到 `careers` 表
- 应看到 `character_careers` 表  
- `characters` 表应包含新列：
  - `main_career_id` (UUID)
  - `main_career_stage` (String, 可空)
  - `sub_careers` (JSON, 可空)

### 步骤5：重启应用
```bash
# 如果使用Docker
docker-compose restart mumuainovel

# 或
docker-compose down && docker-compose up -d
```

---

## 📊 迁移脚本详解

`backend/scripts/create_career_tables.sql` 执行以下操作：

### 创建的表

#### 1. `careers` 表
职业类型定义表，存储项目中定义的所有职业

| 列名 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 主键 |
| `user_id` | String | 用户ID（多用户隔离） |
| `project_id` | UUID | 项目ID外键 |
| `career_type` | String | 职业类型（main/sub） |
| `name` | String | 职业名称 |
| `description` | Text | 职业描述 |
| `stages` | JSON | 职业阶段定义 |
| `created_at` | DateTime | 创建时间 |
| `updated_at` | DateTime | 更新时间 |

#### 2. `character_careers` 表
角色职业关联表，记录每个角色的职业信息和进度

| 列名 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 主键 |
| `user_id` | String | 用户ID（多用户隔离） |
| `character_id` | UUID | 角色ID外键 |
| `career_id` | UUID | 职业ID外键 |
| `current_stage` | String | 当前职业阶段 |
| `progress` | JSON | 职业进度详情 |
| `created_at` | DateTime | 创建时间 |
| `updated_at` | DateTime | 更新时间 |

### 对 `characters` 表的修改

添加三个新列用于快速访问主要职业信息：

| 列名 | 类型 | 说明 |
|------|------|------|
| `main_career_id` | UUID | 主职业ID外键 |
| `main_career_stage` | String | 主职业当前阶段 |
| `sub_careers` | JSON | 副职业ID数组 |

---

## ⚠️ 常见问题

### Q: 迁移会影响现有数据吗？
**A:** 不会。迁移脚本只创建新表和添加新列，不会删除或修改现有数据。新列初始为空/null。

### Q: 如果我遗漏了迁移脚本会怎样？
**A:** 应用会崩溃，错误信息：
```
sqlalchemy.exc.ProgrammingError: column characters.main_career_id does not exist
```
此时需要回到步骤3执行迁移脚本。

### Q: 可以在应用运行时执行迁移吗？
**A:** 可以，但**强烈不推荐**。最佳实践：
1. 停止应用：`docker-compose down`
2. 执行迁移
3. 重启应用：`docker-compose up -d`

### Q: 迁移需要多长时间？
**A:** 通常 < 1秒（对于大多数数据库）。具体时间取决于：
- 数据库大小
- 系统性能
- PostgreSQL配置

### Q: 如何回滚迁移？
**A:** 迁移可以通过以下SQL脚本回滚（**谨慎操作**）：
```sql
-- 删除外键约束
ALTER TABLE character_careers DROP CONSTRAINT IF EXISTS fk_character_id;
ALTER TABLE character_careers DROP CONSTRAINT IF EXISTS fk_career_id;
ALTER TABLE characters DROP CONSTRAINT IF EXISTS fk_main_career_id;

-- 删除新表
DROP TABLE IF EXISTS character_careers CASCADE;
DROP TABLE IF EXISTS careers CASCADE;

-- 删除characters表的新列
ALTER TABLE characters 
  DROP COLUMN IF EXISTS main_career_id,
  DROP COLUMN IF EXISTS main_career_stage,
  DROP COLUMN IF EXISTS sub_careers;
```

---

## 🔧 Docker Compose 用户快速命令

```bash
# 一键迁移（假设容器已运行）
docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel << EOF
\i /scripts/create_career_tables.sql
EOF

# 验证迁移
docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel -c "\d careers"

# 查看characters表结构
docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel -c "\d characters" | grep -E "main_career|sub_careers"
```

---

## 📞 获取帮助

如果迁移过程中遇到问题：

1. **检查数据库连接**
   ```bash
   docker exec mumuainovel-postgres psql -U mumuai -d mumuai_novel -c "SELECT 1"
   ```

2. **查看应用日志**
   ```bash
   docker-compose logs -f mumuainovel
   ```

3. **检查PostgreSQL日志**
   ```bash
   docker-compose logs -f mumuainovel-postgres
   ```

4. **提交Issue**
   包含以下信息：
   - Docker版本
   - PostgreSQL版本
   - 迁移脚本执行输出
   - 应用错误日志

---

## ✅ 迁移检查清单

- [ ] 已备份现有数据库
- [ ] 已拉取最新代码
- [ ] 已停止应用（如果使用Docker）
- [ ] 已执行迁移脚本
- [ ] 已验证新表和新列存在
- [ ] 已重启应用
- [ ] 应用启动无错误
- [ ] 可以访问现有项目和角色
- [ ] 可以创建新角色且包含职业信息
