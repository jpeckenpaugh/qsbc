# Baseline: deepseek-attributable state BEFORE flush/reimport vet
# Captured 2026-09-10T20:49Z, snapshot snapshots/snapshot_20260910-204913.sql.gz

reasonings_total         = 4175
deepseek_reasonings      = 3023
gemma_reasonings         = 1152

deepseek_thoughts        = 11666
deepseek_distinct_texts  = 11666
deepseek_ideas           = 1654   # ideas containing >=1 deepseek thought (mixed clustering)
deepseek_idea_membership = 26088  # sum(i.size) over deepseek-attributable ideas