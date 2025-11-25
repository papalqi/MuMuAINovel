-- 修复 projects 表中 user_id 字段长度不足的问题
-- 将 user_id 从 VARCHAR(36) 扩展到 VARCHAR(100)

ALTER TABLE projects ALTER COLUMN user_id TYPE VARCHAR(100);

-- 验证修改
SELECT column_name, data_type, character_maximum_length 
FROM information_schema.columns 
WHERE table_name = 'projects' AND column_name = 'user_id';