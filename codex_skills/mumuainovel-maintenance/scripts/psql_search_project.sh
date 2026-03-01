#!/usr/bin/env bash
set -euo pipefail

# 在 MuMuAINovel 的关键表中，按 project_id 搜索某个关键词/片段。
#
# 用法：
#   psql_search_project.sh <project_id> <pattern>
#
# 示例：
#   ./psql_search_project.sh 8135... "2035"
#   ./psql_search_project.sh 8135... "三十四岁"
#
# 说明：
# - 在 docker compose 的 postgres 服务容器内执行。
# - 默认账号/库名与 MuMuAINovel 的 docker-compose.yml 保持一致。

PROJECT_ID="${1:?需要 project_id}"
PATTERN="${2:?需要 pattern}"

DB_USER="${DB_USER:-mumuai}"
DB_NAME="${DB_NAME:-mumuai_novel}"
DB_PASS="${DB_PASS:-123456}"

docker compose exec -T postgres bash -lc "PGPASSWORD='${DB_PASS}' psql -U '${DB_USER}' -d '${DB_NAME}' \
  -v pid='${PROJECT_ID}' -v pat='%${PATTERN}%' -c \"
\\echo '== projects.description =='
select id from projects where id=:'pid' and description like :'pat';

\\echo '== outlines.content/structure =='
select id from outlines where project_id=:'pid' and (content like :'pat' or structure::text like :'pat');

\\echo '== chapters.content/summary =='
select chapter_number,id from chapters where project_id=:'pid' and (content like :'pat' or summary like :'pat') order by chapter_number;

\\echo '== characters.background/personality =='
select id,name from characters where project_id=:'pid' and (background like :'pat' or personality like :'pat');

\\echo '== story_memories.content =='
select id from story_memories where project_id=:'pid' and content like :'pat';

\\echo '== plot_analysis (analysis_report + json fields) =='
select chapter_id from plot_analysis where project_id=:'pid' and (
  analysis_report like :'pat'
  or hooks::text like :'pat'
  or plot_points::text like :'pat'
  or foreshadows::text like :'pat'
  or character_states::text like :'pat'
);
\""
