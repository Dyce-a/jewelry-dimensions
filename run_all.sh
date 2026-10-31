#!/usr/bin/env bash
# Полный прогон выборки партиями: каждая партия пишется в таблицу отдельно,
# поэтому обрыв на середине не теряет уже сделанную работу.
set -u
TASK="данные/Рабочая_таблица.xlsx"
mkdir -p "out/_реестр"
for n in 01 02 03 04 05 06 07 08 09 10; do
  batch="Партия $n"
  echo "=== $batch ==="
  python annotate.py --task "$TASK" --batch "$batch" --workers 6 \
    --write-xlsx "$TASK" --csv "out/_реестр/Партия_$n.csv" 2>&1 | tail -14
done
echo "=== готово ==="
