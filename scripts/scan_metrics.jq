# The cargo-geiger JSON SafetyReport stores five categories under unsafety.used.
def geiger_rows:
  if (.packages | type) != "array" or (.packages | length) == 0
     or (.packages_without_metrics | length) > 0 or (.used_but_not_scanned_files | length) > 0
  then error("incomplete cargo-geiger report") else . end
  | [.packages[] | .package.id.name as $name
     | [.unsafety.used | .functions, .exprs, .item_impls, .item_traits, .methods | .unsafe_] as $counts
     | if ($name | type) != "string" or any($counts[]; type != "number" or . < 0)
       then error("invalid cargo-geiger counters") else {name: $name, count: ($counts | add)} end];

def analyzed_findings:
  if (.results | type) == "array" and (.errors | length) == 0 then .results
  else error("incomplete OpenGrep report") end;
